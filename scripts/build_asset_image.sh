#!/usr/bin/env bash
# Build the figure-and-table image and push it to this stack's ECR repository.
#
# The image carries marker and its two models so that a paper arriving on its own converts in
# seconds instead of waiting three minutes for a pip install. That is the only reason it exists,
# and it is rebuilt only when marker or its models change - the worker code itself is read from
# s3://{bucket}/worker/assets.py at start, so fixing the extraction needs no rebuild.
#
# Architecture is ARM64, matching the task definition: the work is pure CPU, this builds natively
# on an Apple Silicon Mac with no emulation, and Graviton costs about a fifth less to run.
#
#   scripts/build_asset_image.sh              build, push, and publish the worker
#   scripts/build_asset_image.sh --worker     publish only the worker code (no image build)
#
# Requires a running container runtime (`colima start`) and a sourced .byeori.env.
set -euo pipefail

cd "$(dirname "$0")/.."

: "${AWS_KIRO_WIKI_BUCKET:?source .byeori.env first}"
: "${KIRO_WIKI_STACK:?source .byeori.env first}"
REGION="${AWS_REGION:?set AWS_REGION (source the env file first)}"
TAG="${ASSET_IMAGE_TAG:-latest}"

publish_worker() {
  echo "==> publishing infra/asset_worker.py to s3://$AWS_KIRO_WIKI_BUCKET/worker/assets.py"
  aws s3 cp infra/asset_worker.py "s3://$AWS_KIRO_WIKI_BUCKET/worker/assets.py" --only-show-errors
}

if [ "${1:-}" = "--worker" ]; then
  publish_worker
  echo "done: the next task will pick it up; no image was built."
  exit 0
fi

REPO_URI="$(aws cloudformation describe-stacks --stack-name "$KIRO_WIKI_STACK" --region "$REGION" \
  --query "Stacks[0].Outputs[?OutputKey=='AssetRepositoryUri'].OutputValue" --output text)"
if [ -z "$REPO_URI" ] || [ "$REPO_URI" = "None" ]; then
  echo "the stack has no AssetRepositoryUri output; deploy infra/template.yaml first" >&2
  exit 1
fi

echo "==> building linux/arm64 image (the build converts a PDF; it fails here if a model is missing)"
docker build -f infra/asset-worker.Dockerfile -t "byeori-asset-worker:$TAG" .

echo "==> signing in to $REPO_URI"
aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "${REPO_URI%%/*}"

echo "==> pushing $REPO_URI:$TAG"
docker tag "byeori-asset-worker:$TAG" "$REPO_URI:$TAG"
docker push "$REPO_URI:$TAG"

publish_worker

echo
echo "done. A paper stored from now on has its figures cut without anything else being run."
echo "To redo a paper that is already here:"
echo "  aws ecs run-task --cluster <ExtractionClusterName> --task-definition <AssetTaskDefinitionArn> \\"
echo "    --launch-type FARGATE --network-configuration ... \\"
echo "    --overrides '{\"containerOverrides\":[{\"name\":\"worker\",\"environment\":[{\"name\":\"STEMS\",\"value\":\"<stem>\"},{\"name\":\"FORCE\",\"value\":\"1\"}]}]}'"
