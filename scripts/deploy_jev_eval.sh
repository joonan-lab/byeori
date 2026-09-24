#!/bin/bash
# Deploy the isolated Jev evaluation stack (infra/jev-eval-template.json) as `byeori-jev-eval`.
#
# The package is not src/ as a whole. The template's handler is `index.handler`, so the worker
# module ships as index.py at the zip root with only the modules it imports beside it. Packaging
# all of src/ broke the function on 2026-09-22: byeori.ingest_lambda reads BUCKET_NAME at
# import time, which the evaluation stack does not set, so every invocation failed at import.
#
# Usage: source .byeori.env; bash scripts/deploy_jev_eval.sh
# Required: AWS_KIRO_WIKI_BUCKET, LAB_JEV_KMS_KEY_ARN. Optional: JEV_EVAL_STACK (default
# byeori-jev-eval), JEV_API_KEY_PARAMETER (default /byeori/jev/api-key).
set -euo pipefail

: "${AWS_KIRO_WIKI_BUCKET:?source .byeori.env first}"
: "${LAB_JEV_KMS_KEY_ARN:?set LAB_JEV_KMS_KEY_ARN to the KMS key ARN that encrypts the Jev parameter}"
stack="${JEV_EVAL_STACK:-byeori-jev-eval}"
parameter="${JEV_API_KEY_PARAMETER:-/byeori/jev/api-key}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Modules the worker imports, directly or through jev_triage and evidence_packet.
modules=(__init__ jev_triage wiki_search evidence_packet lab_policy wiki_connections)

workdir="$(mktemp -d)"
trap 'rm -rf "$workdir"' EXIT
mkdir -p "$workdir/pkg/byeori"
cp "$root/src/byeori/jev_eval.py" "$workdir/pkg/index.py"
for module in "${modules[@]}"; do
  cp "$root/src/byeori/$module.py" "$workdir/pkg/byeori/"
done
(cd "$workdir/pkg" && find . -name '*.py' | sort | zip -q -X "$workdir/jev.zip" -@)
digest="$(shasum -a 256 "$workdir/jev.zip" | cut -d' ' -f1)"
key="runs/jev-evaluations/deployments/$digest.zip"
aws s3 cp "$workdir/jev.zip" "s3://$AWS_KIRO_WIKI_BUCKET/$key" --only-show-errors
printf 'Uploaded s3://%s/%s\n' "$AWS_KIRO_WIKI_BUCKET" "$key"

aws cloudformation deploy \
  --template-file "$root/infra/jev-eval-template.json" \
  --stack-name "$stack" \
  --capabilities CAPABILITY_IAM \
  --no-fail-on-empty-changeset \
  --parameter-overrides \
    "ArtifactBucket=$AWS_KIRO_WIKI_BUCKET" \
    "ArtifactKey=$key" \
    "ResultsBucket=$AWS_KIRO_WIKI_BUCKET" \
    "JevApiKeyParameter=$parameter" \
    "ParameterKmsKeyArn=$LAB_JEV_KMS_KEY_ARN"

aws lambda get-function-configuration --function-name "$stack" \
  --query '[LastUpdateStatus,State,Runtime,Timeout,MemorySize]' --output text
