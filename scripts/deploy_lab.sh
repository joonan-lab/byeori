#!/bin/bash
# Deploy the separate student stack (infra/lab-template.json) as `byeori-lab`.
#
# Written but not executed on 2026-09-21: the question campaign was still running in the main stack.
# Run only after the question campaign is finished. The script refuses to proceed while an
# execution of the question state machine is RUNNING unless --allow-during-campaign is passed, and
# it only creates and prints a change set (--no-execute-changeset) unless --execute is passed.
#
# Usage: source .byeori.env; bash scripts/deploy_lab.sh [--execute] [--allow-during-campaign]
# Required: AWS_KIRO_WIKI_BUCKET (data and artifact bucket), LAB_JEV_KMS_KEY_ARN (KMS key of the Jev
# SecureString). Optional: LAB_STACK (default byeori-lab), QUESTION_STATE_MACHINE_ARN (otherwise read
# from the KIRO_WIKI_STACK output QuestionStateMachineArn).
# Stack settings are changed only by naming them: LAB_ANSWER_MODEL_ID, LAB_ANSWER_REASONING,
# LAB_POLICY_REVISION, LAB_RESEARCH_CONSUMER_ENABLED (true connects the approved research consumer),
# LAB_RESEARCH_MODEL_ID, LAB_RESEARCH_REASONING, LAB_OUTBOX_SCHEDULE_ENABLED (false parks the
# 1-minute relay). Each one that is unset keeps the value the stack already has, and the script
# prints which ones it carried over. Uploads go under cfn/lab/ only; nothing is deleted or
# synchronised.
set -euo pipefail

execute=0
allow_during_campaign=0
for argument in "$@"; do
  case "$argument" in
    --execute) execute=1 ;;
    --allow-during-campaign) allow_during_campaign=1 ;;
    -h|--help) sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) printf 'Unknown argument: %s\n' "$argument" >&2; exit 2 ;;
  esac
done

: "${AWS_KIRO_WIKI_BUCKET:?source .byeori.env first}"
: "${LAB_JEV_KMS_KEY_ARN:?set LAB_JEV_KMS_KEY_ARN to the KMS key ARN that encrypts /byeori/jev/api-key}"
stack="${LAB_STACK:-byeori-lab}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
template="$root/infra/lab-template.json"

# 1. Campaign guard. The new stack shares the account's Lambda concurrency and the data bucket with
#    the running question campaign, so nothing is uploaded or deployed while an execution is RUNNING.
machine="${QUESTION_STATE_MACHINE_ARN:-}"
if [[ -z "$machine" ]]; then
  : "${KIRO_WIKI_STACK:?set QUESTION_STATE_MACHINE_ARN or KIRO_WIKI_STACK to locate the question state machine}"
  machine="$(aws cloudformation describe-stacks --stack-name "$KIRO_WIKI_STACK" \
    --query "Stacks[0].Outputs[?OutputKey=='QuestionStateMachineArn'].OutputValue" --output text)"
fi
if [[ -z "$machine" || "$machine" == "None" ]]; then
  printf 'Could not resolve the question state machine ARN; refusing to deploy.\n' >&2
  exit 1
fi
running="$(aws stepfunctions list-executions --state-machine-arn "$machine" --status-filter RUNNING \
  --max-results 5 --query 'executions[].executionArn' --output text)"
if [[ -n "$running" && "$running" != "None" ]]; then
  if [[ "$allow_during_campaign" -ne 1 ]]; then
    printf 'A question campaign execution is RUNNING; refusing to deploy:\n%s\n' "$running" >&2
    printf 'Pass --allow-during-campaign only when the user has accepted deploying during the campaign.\n' >&2
    exit 1
  fi
  printf 'Warning: deploying while a question campaign execution is RUNNING (--allow-during-campaign).\n' >&2
fi

# 2. Package src/ (*.py and *.json) and upload it under cfn/lab/<sha256>.zip. The zip root is src/, so the
#    handler module path byeori.lab_lambda is importable.
workdir="$(mktemp -d)"
trap 'rm -rf "$workdir"' EXIT
(cd "$root/src" && find . \( -name '*.py' -o -name '*.json' \) -not -path '*/__pycache__/*' | sort \
  | zip -q -X "$workdir/lab.zip" -@)
digest="$(shasum -a 256 "$workdir/lab.zip" | cut -d' ' -f1)"
artifact_key="cfn/lab/$digest.zip"
aws s3 cp "$workdir/lab.zip" "s3://$AWS_KIRO_WIKI_BUCKET/$artifact_key" --only-show-errors
printf 'Uploaded s3://%s/%s\n' "$AWS_KIRO_WIKI_BUCKET" "$artifact_key"

# 3. Validate the template through S3 (the body limit of validate-template is 51,200 bytes).
template_key="cfn/lab/$digest-template.json"
aws s3 cp "$template" "s3://$AWS_KIRO_WIKI_BUCKET/$template_key" --only-show-errors
bucket_region="$(aws s3api get-bucket-location --bucket "$AWS_KIRO_WIKI_BUCKET" \
  --query LocationConstraint --output text)"
if [[ -z "$bucket_region" || "$bucket_region" == "None" ]]; then
  bucket_region="us-east-1"
fi
aws cloudformation validate-template \
  --template-url "https://$AWS_KIRO_WIKI_BUCKET.s3.$bucket_region.amazonaws.com/$template_key" >/dev/null
printf 'Template validated.\n'

# 4. Create the change set without executing it (the default), then print it for review.
#    Only the four values this build actually determines are sent. A setting whose variable is
#    unset in this shell is left out, and `aws cloudformation deploy` then marks it
#    UsePreviousValue, so the stack keeps what it already has. Sending a shell default instead
#    would silently undo a setting somebody turned on: LAB_RESEARCH_CONSUMER_ENABLED defaulted to
#    false here while the deployed stack had it true, so any deploy that only shipped new code
#    would have detached the approved-research queue without saying so. On the first create there
#    is no previous value and the template's own Default applies.
parameters=(
  "DataBucket=$AWS_KIRO_WIKI_BUCKET"
  "ArtifactBucket=$AWS_KIRO_WIKI_BUCKET"
  "ArtifactKey=$artifact_key"
  "ParameterKmsKeyArn=$LAB_JEV_KMS_KEY_ARN"
)
carry_over=()
add_optional() {
  local variable="$1" parameter="$2"
  if [[ -n "${!variable:-}" ]]; then
    parameters+=("$parameter=${!variable}")
  else
    carry_over+=("$parameter")
  fi
}
add_optional LAB_ANSWER_MODEL_ID AnswerModelId
add_optional LAB_ANSWER_REASONING AnswerReasoning
add_optional LAB_POLICY_REVISION PolicyRevision
add_optional LAB_RESEARCH_CONSUMER_ENABLED ResearchConsumerEnabled
add_optional LAB_RESEARCH_MODEL_ID ResearchModelId
add_optional LAB_RESEARCH_REASONING ResearchReasoning
add_optional LAB_OUTBOX_SCHEDULE_ENABLED OutboxScheduleEnabled
if [[ "${#carry_over[@]}" -gt 0 ]]; then
  printf 'Keeping the deployed value of: %s\n' "${carry_over[*]}"
fi
status="$(aws cloudformation describe-stacks --stack-name "$stack" \
  --query 'Stacks[0].StackStatus' --output text 2>/dev/null || true)"
if [[ -z "$status" || "$status" == "REVIEW_IN_PROGRESS" ]]; then
  wait_condition=stack-create-complete
else
  wait_condition=stack-update-complete
fi
deploy_output="$(aws cloudformation deploy \
  --template-file "$template" \
  --s3-bucket "$AWS_KIRO_WIKI_BUCKET" --s3-prefix cfn/lab \
  --stack-name "$stack" \
  --capabilities CAPABILITY_IAM \
  --no-execute-changeset --no-fail-on-empty-changeset \
  --parameter-overrides "${parameters[@]}")"
printf '%s\n' "$deploy_output"
change_set="$(printf '%s\n' "$deploy_output" \
  | grep -o 'arn:aws[a-z-]*:cloudformation:[^ ]*changeSet/[^ ]*' | tail -n1 || true)"
if [[ -z "$change_set" ]]; then
  printf 'No change set was created (nothing to deploy).\n'
  exit 0
fi
aws cloudformation describe-change-set --change-set-name "$change_set" \
  --query 'Changes[].ResourceChange.[Action,LogicalResourceId,ResourceType,Replacement]' --output table
if [[ "$execute" -ne 1 ]]; then
  printf 'Change set created but not executed. Review it, then re-run with --execute.\n'
  exit 0
fi

# 5. Execute only on request, wait for the stack, and print the outputs the clients need.
aws cloudformation execute-change-set --change-set-name "$change_set"
aws cloudformation wait "$wait_condition" --stack-name "$stack"
aws cloudformation describe-stacks --stack-name "$stack" \
  --query "Stacks[0].Outputs[?OutputKey=='GatewayUrl' || OutputKey=='ControlTableName' || OutputKey=='StudentAccessPolicyArn' || OutputKey=='AdminAccessPolicyArn'].[OutputKey,OutputValue]" \
  --output table
