# Devin Review

Trigger or check a pull/merge request review using the organization-scoped Devin Review v3 API:

```bash
export LLM_DEVIN_ORG_ID=your_org_id
export LLM_DEVIN_KEY=your_api_key
llm devin review start https://github.com/owner/repo/pull/123
llm devin review status https://github.com/owner/repo/pull/123
llm devin review status https://github.com/owner/repo/pull/123 --commit abcd1234 --json
```

Keys can also be stored with `llm keys set devin` or supplied with `--key`.
GitHub pull requests, GitLab merge requests (including subgroups), and enterprise/self-managed hosts are supported. URLs must use HTTP or HTTPS. Commit values must be hexadecimal, 4–64 characters.

## Permissions and behavior

`start` requires **UseReviewManual** and sends one POST without retries. It always requests a review of the latest PR head commit; it cannot target an older commit.

`status` requires **ViewReview** and sends only a GET. It never triggers a review. Without `--commit`, the provider resolves the PR's current head. With `--commit`, it looks up the supplied full SHA or short prefix. Commands fetch once and exit, without polling.

A status 404 means no review exists for the resolved commit, not necessarily that the PR is missing. A new push can therefore cause a 404 even if an older commit was reviewed. Use `llm devin review start PR_URL` to request a review of the current head. A start 404 means the PR was not found or is not accessible to Devin.

## Output and statuses

Human output includes the input PR URL, repository path, PR number, commit SHA, status, and creation timestamp. Documented statuses are `pending`, `running`, `completed`, `errored`, `cancelled`, and `skipped`. Pending/running indicate work is not finished; the other documented statuses are terminal. Unknown future status strings are displayed unchanged.

The API returns **status only**, not findings or a review URL. Findings are posted on the PR itself; open the PR to read them.

Both commands accept `--json`, producing only `{"pr_url": "...", "review": {...}}` on stdout. Errors go to stderr with a non-zero exit status. Authentication, permissions, conflicts, validation failures, rate limits, and network failures are reported without retrying.
