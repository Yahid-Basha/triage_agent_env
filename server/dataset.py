# weight field controls sampling probability — hard samples are favoured 2x over easy
TASKS = {
    "relevance": [
        # --- easy ---
        {
            "query": "How do I reset my enterprise portal password?",
            "chunks": [
                "To reset your password, visit the IT portal and click 'Forgot Password'.",
                "The quarterly earnings report showed 12% growth in Q3.",
                "Password policies require minimum 8 characters with special symbols."
            ],
            "relevant_ids": [0, 2],
            "difficulty": "easy",
            "weight": 1
        },
        {
            "query": "What is the SLA for P1 incidents?",
            "chunks": [
                "P1 incidents must be acknowledged within 15 minutes and resolved within 4 hours.",
                "The cafeteria menu changes every Monday.",
                "Incident priority is determined by business impact and affected user count.",
                "All SLA breaches must be reported to the on-call manager immediately."
            ],
            "relevant_ids": [0, 2, 3],
            "difficulty": "easy",
            "weight": 1
        },
        # --- medium ---
        {
            "query": "How do I request access to the production database?",
            "chunks": [
                "Production database access requires approval from your team lead and the DBA team.",
                "Submit a ServiceNow ticket under category 'Access Management > Database Access'.",
                "The company picnic is scheduled for June 15th at Riverside Park.",
                "Access is provisioned within 2 business days after all approvals are collected.",
                "All production access is subject to quarterly access reviews and auto-revoked after 90 days."
            ],
            "relevant_ids": [0, 1, 3, 4],
            "difficulty": "medium",
            "weight": 1
        },
        {
            "query": "What is the process for onboarding a new vendor integration?",
            "chunks": [
                "Vendor onboarding requires a security assessment completed by the InfoSec team.",
                "The break room microwave was replaced last Thursday.",
                "New API integrations must pass a penetration test before connecting to production.",
                "Vendors must sign a Data Processing Agreement (DPA) before data exchange begins.",
                "Integration patterns are documented in the internal developer wiki under 'Third-Party Integrations'.",
                "Contact HR for employee benefit enrollment details."
            ],
            "relevant_ids": [0, 2, 3, 4],
            "difficulty": "hard",
            "weight": 2
        },
        # --- hard: topically-related distractors ---
        {
            # Chunks 1 and 3 are about SSO/Okta but answer a DIFFERENT question (generic SSO overview
            # vs. the specific Okta SCIM provisioning setup being asked). Frontier models often
            # treat "related domain" as "relevant".
            "query": "How do I configure SCIM provisioning for Okta in our identity platform?",
            "chunks": [
                "SCIM provisioning for Okta requires enabling the SCIM 2.0 connector in the Okta Admin console under Directory > Provisioning.",
                "Single Sign-On (SSO) lets users authenticate once and access multiple apps without re-entering credentials.",
                "To activate SCIM, generate a Bearer token in the identity platform and paste it into Okta's provisioning settings.",
                "Okta supports SAML 2.0 and OIDC protocols for federated authentication across enterprise applications.",
                "After enabling SCIM, map attributes: userName → email, givenName → first_name, familyName → last_name.",
                "SSO reduces password fatigue and is recommended by NIST SP 800-63B as a best practice."
            ],
            # Chunks 1, 3, 5 are about SSO/Okta in general but do NOT answer the SCIM config question
            "relevant_ids": [0, 2, 4],
            "difficulty": "hard",
            "weight": 2
        },
        {
            # Chunks mention "API versioning" broadly; only the ones specific to v2.3 are relevant.
            # A distracted model picks all API-related chunks.
            "query": "What breaking changes were introduced in the internal REST API v2.3?",
            "chunks": [
                "API v2.3 removed the deprecated /users/search endpoint; use /users/query with POST instead.",
                "REST API design principles recommend using nouns for resource names and HTTP verbs for actions.",
                "The /auth/token endpoint in v2.3 now requires the client_id field in the request body.",
                "API versioning strategies include URI versioning, header versioning, and query-parameter versioning.",
                "Pagination in v2.3 changed from offset/limit to cursor-based; the next_cursor field replaces the page parameter.",
                "Always validate API responses against the published JSON schema to catch contract violations early."
            ],
            "relevant_ids": [0, 2, 4],
            "difficulty": "hard",
            "weight": 2
        },
    ],
    "hallucination": [
        # --- medium ---
        {
            "query": "What caused the payment service outage on March 3rd?",
            "context": "On March 3rd, the payment service went down due to a database connection pool exhaustion. The issue was resolved by increasing the pool size from 50 to 200 connections. Recovery took 2.5 hours.",
            "answer": "The payment service outage on March 3rd was caused by a database connection pool exhaustion. Engineers resolved it by increasing the pool size from 50 to 500 connections. The service was restored within 1 hour.",
            # Injected hallucinations: "500" (should be 200), "1 hour" (should be 2.5 hours)
            "hallucinations": ["500 connections", "1 hour"],
            "difficulty": "medium",
            "weight": 1
        },
        {
            "query": "What are the data retention policies for support tickets?",
            "context": "Support tickets are retained for 7 years per regulatory requirements. Tickets containing PII are anonymized after 2 years. Archived tickets are stored in cold storage on AWS S3 Glacier.",
            "answer": "Support tickets are kept for 5 years according to company policy. Tickets with personal data are anonymized after 2 years. Archives are stored in AWS S3 Standard storage for cost efficiency.",
            # Injected hallucinations: "5 years" (should be 7), "S3 Standard" (should be S3 Glacier)
            "hallucinations": ["5 years", "S3 Standard"],
            "difficulty": "medium",
            "weight": 1
        },
        # --- hard (obvious multi-error) ---
        {
            "query": "How does the auto-scaling policy work for the API gateway?",
            "context": "The API gateway scales out when CPU utilization exceeds 70% for 3 consecutive minutes. It scales in when CPU drops below 30% for 10 minutes. Maximum instance count is capped at 20. Scaling events are logged to CloudWatch.",
            "answer": "The API gateway automatically adds instances when CPU usage goes above 70% for 3 consecutive minutes. It scales in when CPU falls below 30% for 5 minutes. The maximum number of instances is 50. Scaling activity is recorded in CloudTrail.",
            # Hallucinations: "5 minutes" (should be 10), "50 instances" (should be 20), "CloudTrail" (should be CloudWatch)
            "hallucinations": ["5 minutes", "50 instances", "CloudTrail"],
            "difficulty": "hard",
            "weight": 2
        },
        # --- hard: subtle single-digit / near-miss hallucinations ---
        {
            # Cache hit rate is off by 1 percentage point; the second error swaps two field names.
            # Models that skim numbers often miss these.
            "query": "What were the results of the embedding cache optimisation released in sprint 41?",
            "context": "Sprint 41 shipped an embedding cache that raised the cache hit rate from 67% to 78%, cutting average retrieval latency from 340 ms to 95 ms. The cache uses an LRU eviction policy with a TTL of 24 hours.",
            "answer": "The sprint 41 embedding cache raised the hit rate from 68% to 78%, reducing retrieval latency from 340 ms to 95 ms. It uses an LRU eviction policy with a TTL of 24 hours.",
            # Only one subtle hallucination: starting hit rate is 68% not 67%
            "hallucinations": ["68%"],
            "difficulty": "hard",
            "weight": 2
        },
        {
            # The maintenance window hour is shifted by one; easy to miss when numbers are plausible.
            "query": "When is the scheduled database maintenance window and what will be affected?",
            "context": "The database maintenance window is Saturday 11:00 PM to Sunday 3:00 AM UTC. During this window, read replicas will remain available but all write operations will be queued. Expected downtime for write paths is under 15 minutes.",
            "answer": "The database maintenance runs from Saturday 10:00 PM to Sunday 3:00 AM UTC. Read replicas stay online while writes are queued. Write path downtime is expected to be under 15 minutes.",
            # Hallucination: start time is 10 PM not 11 PM
            "hallucinations": ["10:00 PM"],
            "difficulty": "hard",
            "weight": 2
        },
    ],
    "full_judgment": [
        # --- easy ---
        {
            "query": "What is the process for handling a data breach incident?",
            "chunks": [
                "Upon detecting a data breach, the security team must be notified within 1 hour.",
                "The incident response team follows the NIST 800-61 framework for breach containment.",
                "Affected customers must be notified within 72 hours per GDPR requirements.",
                "A post-incident review must be completed within 5 business days."
            ],
            "answer": "When a data breach is detected, the security team is notified within 1 hour. The IR team follows NIST 800-61 for containment. Affected customers are notified within 72 hours per GDPR. A post-incident report is completed within 5 business days.",
            "cited_ids": [0, 1, 2, 3],
            "ground_truth": {
                "relevance": 1.0,
                "faithfulness": 1.0,
                "citation_accuracy": 1.0
            },
            "difficulty": "easy",
            "weight": 1
        },
        # --- medium ---
        {
            "query": "How does the RAG pipeline handle low-confidence retrievals?",
            "chunks": [
                "Retrievals with similarity scores below 0.6 are flagged as low-confidence.",
                "Low-confidence chunks trigger a fallback to keyword-based BM25 search.",
                "The system logs all fallback events to the observability dashboard.",
                "Engineers review fallback patterns weekly to improve embedding quality."
            ],
            "answer": "The pipeline flags retrievals with similarity below 0.7 as low-confidence. These trigger a fallback to BM25 keyword search. All such events are logged for observability.",
            "cited_ids": [0, 1, 2],
            "ground_truth": {
                "relevance": 1.0,
                "faithfulness": 0.7,
                "citation_accuracy": 0.85
            },
            "difficulty": "medium",
            "weight": 1
        },
        # --- hard ---
        {
            "query": "Explain the ticket deduplication logic in the RAG system.",
            "chunks": [
                "Duplicate tickets are identified using cosine similarity above 0.85 threshold.",
                "The system uses ChromaDB for vector storage with sentence-transformers embeddings.",
                "Deduplication runs every 30 minutes on the ingestion pipeline."
            ],
            "answer": "The RAG system identifies duplicate tickets by computing cosine similarity, flagging pairs above 0.85. It uses ChromaDB with sentence-transformers. Deduplication runs hourly.",
            "cited_ids": [0, 1],
            # Answer is faithful except "hourly" should be "every 30 minutes"; chunk 2 used but not cited
            "ground_truth": {
                "relevance": 1.0,
                "faithfulness": 0.6,
                "citation_accuracy": 0.8
            },
            "difficulty": "hard",
            "weight": 2
        },
        {
            # Adversarial: the answer cites [0,1,2,3] but only uses information from [0,2].
            # Chunks 1 and 3 are about monitoring/alerting (related domain) but NOT about
            # the reranking logic being asked. A model that simply checks "are these chunks
            # in the same domain?" will over-score citation_accuracy.
            "query": "How does the reranking step work in the enterprise knowledge retrieval pipeline?",
            "chunks": [
                "The reranker uses a cross-encoder model (ms-marco-MiniLM-L-6-v2) to score query-chunk pairs.",
                "The Prometheus metrics dashboard tracks retrieval latency at p50, p95, and p99 percentiles.",
                "Top-k chunks from the bi-encoder are re-scored; only chunks above 0.4 cross-encoder score are passed to the LLM.",
                "Alerting rules fire a PagerDuty notification when p99 latency exceeds 800 ms for 5 consecutive minutes."
            ],
            "answer": "The reranking step uses a cross-encoder (ms-marco-MiniLM-L-6-v2) to score each query-chunk pair. Only chunks with a cross-encoder score above 0.4 are forwarded to the LLM. Latency is tracked at p50, p95, and p99 via Prometheus.",
            "cited_ids": [0, 1, 2, 3],
            # Chunks 1 and 3 (monitoring/alerting) are NOT relevant to the reranking question.
            # The answer faithfully describes reranking (from 0,2) but falsely cites 1 and 3.
            "ground_truth": {
                "relevance": 0.6,    # only chunks 0 and 2 are truly relevant
                "faithfulness": 0.85, # reranking facts are correct; latency detail is tangential
                "citation_accuracy": 0.6  # chunks 1 & 3 cited but don't support the reranking claims
            },
            "difficulty": "hard",
            "weight": 2
        },
        {
            # Adversarial: answer paraphrases all chunks faithfully but slightly misattributes
            # which chunk supports which claim (cited_ids are shuffled).
            "query": "What embedding model and indexing strategy does the support-ticket RAG system use?",
            "chunks": [
                "The system embeds support tickets using the all-mpnet-base-v2 model from sentence-transformers.",
                "Embeddings are indexed in a HNSW graph structure with ef_construction=200 and M=16.",
                "The index is rebuilt nightly at 2:00 AM UTC to incorporate newly closed tickets.",
                "Approximate nearest-neighbour search with HNSW reduces query latency by 60% versus brute-force."
            ],
            "answer": "Support tickets are embedded with all-mpnet-base-v2 and indexed using HNSW (ef_construction=200, M=16). The index rebuilds nightly at 2 AM UTC. HNSW approximate search cuts query latency by 60% compared to brute-force.",
            "cited_ids": [0, 1, 2, 3],
            # All four chunks are relevant and fully used — perfect answer
            "ground_truth": {
                "relevance": 1.0,
                "faithfulness": 1.0,
                "citation_accuracy": 1.0
            },
            "difficulty": "medium",
            "weight": 1
        },
    ]
}
