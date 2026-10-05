# Code signing

Release installers are built by `.github/workflows/release.yml`. Until signing is configured
they ship **unsigned**: they work, but Windows SmartScreen shows "Windows protected your PC"
and users must click *More info -> Run anyway*. Verify an unsigned download against
`SHA256SUMS.txt` on the release page.

## Turning signing on (Azure Artifact Signing, ~US$10/month)

1. Azure portal (paid subscription, not free/trial): create an **Artifact Signing account**.
2. Complete **identity validation** (individuals: USA/Canada; organizations: also EU/UK).
3. Create a **Public Trust certificate profile**.
4. Create an app registration (service principal) and give it the
   *Artifact Signing Certificate Profile Signer* role on the account.
5. Add these repository secrets (Settings -> Secrets and variables -> Actions):
   `AZURE_TENANT_ID`, `AZURE_CLIENT_ID`, `AZURE_CLIENT_SECRET`,
   `SIGNING_ENDPOINT` (e.g. `https://eus.codesigning.azure.net/`),
   `SIGNING_ACCOUNT`, `SIGNING_PROFILE`.

The next tagged release is signed automatically; nothing else changes.

## Alternative: SignPath Foundation (free for open source)

Requires an OSI-approved license with no proprietary components; the publisher shown to
users is "SignPath Foundation". Apply at https://signpath.org once a LICENSE is in place.
