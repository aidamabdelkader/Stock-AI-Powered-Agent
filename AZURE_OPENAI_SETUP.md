# Azure OpenAI setup

1. Copy the Azure environment template:

```bash
cp .env.azure.example .env
```

2. Update these four values in `.env`:

```dotenv
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=https://YOUR-RESOURCE-NAME.openai.azure.com/
AZURE_OPENAI_API_VERSION=2025-04-01-preview
AZURE_OPENAI_DEPLOYMENT=YOUR-DEPLOYMENT-NAME
```

`AZURE_OPENAI_DEPLOYMENT` must be the deployment name created in Azure. It may be different from the underlying model name.

3. Install the full dependencies and index the articles:

```bash
./scripts/bootstrap.sh
make index
```

4. Start the API:

```bash
make api
```

5. Verify the provider:

```bash
curl http://localhost:8000/health
```

Expected fields:

```json
{
  "generation_ready": true,
  "llm_provider": "azure_openai",
  "model": "YOUR-DEPLOYMENT-NAME"
}
```

6. Test an answer:

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Why did banking stocks lead the EGX30 gains?",
    "debug": true
  }'
```
