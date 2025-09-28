$headers = @{
		"Content-Type" = "application/json"
		"Authorization" = "Bearer $env:OPENAI_API_KEY"
}

$body = @{
		"messages" = @(
				@{
						"role" = "system"
						"content" = "You are a test assistant."
				},
				@{
						"role" = "user"
						"content" = "Testing. Just say hi and nothing else."
				}
		)
		"model" = "gpt-4o-mini"
} | ConvertTo-Json

Invoke-WebRequest -Uri "https://api.openai.com/v1/chat/completions" -Method Post -Headers $headers -Body $body