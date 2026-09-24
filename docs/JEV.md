# Jev triage (optional)

## What Jev does here

Jev is a model from TypeSafe that answers a structured question with a probability. Byeori uses
it in one place: after the student service (`docs/LAB-SERVICE.md`) has answered a student's
question, the triage worker asks Jev whether the question would deserve a new or improved
synthesis page. When the probability is at least 0.99 and a concrete gap in the wiki was found,
the answer carries a synthesis offer, which runs only if the student accepts it.

Jev never sees a paper's full text. It receives the question and short, bounded excerpts, at most
30,000 bytes per call, from inside AWS. Questions marked as containing unpublished material are
not sent at all. If Jev fails or has no key, the student still gets the answer; only the offer is
missing. The key is read in AWS by the triage worker and nowhere else.

The call goes to TypeSafe's SystemOne endpoint, `https://api.typesafe.ai/v1/systemone`, with the
key as a Bearer token.

## Cost

- **Jev itself**: a fraction of a cent per answer. Byeori records an estimate at $0.042 per million
  input tokens; one lab's twelve-question trial came to about $0.0018 in total. TypeSafe's own
  pricing decides the invoice; check it on their site.
- **The KMS key** that encrypts the Jev key: $1 per month. The student service needs this key even
  without Jev (`docs/INSTALL.md` step 7), so Jev adds nothing here if you already made it.
- **Parameter Store**: free.

## Get a key from TypeSafe

Create an account at [typesafe.ai](https://typesafe.ai) and issue an API key from your account
page. TypeSafe's quickstart says to get the key from the dashboard at
https://console.typesafe.ai/keys; the exact labels are in TypeSafe's documentation,
https://docs.typesafe.ai. The key is used for the SystemOne endpoint above.

Copy the key once, straight into the command in the next section. Do not paste it into a chat
with an agent, an e-mail or a file.

## Store the key

The key goes into Parameter Store as a `SecureString`, encrypted with a KMS key you own. If you
did not create the key in `docs/INSTALL.md` step 7, create it now:

```bash
source .byeori.env
aws kms create-key --description "byeori parameters"
aws kms create-alias --alias-name alias/byeori-parameters --target-key-id <KeyId from the output>
```

Then store the Jev key under its fixed name, encrypted with that key:

```bash
aws ssm put-parameter --name /byeori/jev/api-key --type SecureString --key-id <key arn> --value <your Jev key>
```

Why the name is fixed: the student stack's `JevApiKeyParameter` accepts only `/byeori/jev/api-key`,
and the triage worker's role may read that one parameter and nothing else. A fixed name means no
setting can point the worker at a different secret.

## Enable it

1. Put the KMS key's ARN in `.byeori.env`, if it is not there from step 7:

   ```bash
   echo "export LAB_JEV_KMS_KEY_ARN=<key arn>" >> .byeori.env
   source .byeori.env
   ```

2. Deploy (or redeploy) the student stack:

   ```bash
   uv run byeori deploy-lab
   ```

   This passes the two template parameters: `JevApiKeyParameter` (always `/byeori/jev/api-key`)
   and `ParameterKmsKeyArn` (your `LAB_JEV_KMS_KEY_ARN`). The triage worker's role is allowed to
   decrypt with exactly that key, and only through Parameter Store for that one parameter.

From the next answer on, triage runs after every answer that is not marked private.

## Confirm it

To check the key without touching the student service, deploy the isolated evaluation stack. It
has one function that can read the Jev parameter and write receipts under
`runs/jev-evaluations/`, and cannot read papers, edit the wiki or call Bedrock:

```bash
source .byeori.env
bash scripts/deploy_jev_eval.sh
```

Then make one call. The function is named after its stack, `byeori-jev-eval`:

```bash
aws lambda invoke --function-name byeori-jev-eval --cli-binary-format raw-in-base64-out \
  --payload '{"action": "smoke"}' jev-smoke.json
cat jev-smoke.json
```

`"status": "ok"` means AWS read the key and Jev accepted it and answered in the expected shape.
The call costs a small fraction of a cent. `{"action": "check_key_format"}` instead checks only
that the stored value looks like a key (no spaces, quotes or a `Bearer` prefix pasted with it),
without calling Jev. Neither result ever contains the key. Delete the stack when you are done:
`aws cloudformation delete-stack --stack-name byeori-jev-eval`.

## Rotate and revoke

To rotate the key, issue a new one at TypeSafe and put it as a new version of the same parameter:

```bash
aws ssm put-parameter --name /byeori/jev/api-key --type SecureString --key-id <key arn> --value <new key> --overwrite
```

The worker reads the parameter on every call and always gets the latest version, so nothing needs
redeploying. Then revoke the old key in TypeSafe's dashboard. If a key may have leaked, revoke it
there first; triage fails safely (answers still arrive, without offers) until the new key is
stored.

To turn Jev off, revoke the key at TypeSafe and delete the parameter
(`aws ssm delete-parameter --name /byeori/jev/api-key`). The student service keeps answering.
