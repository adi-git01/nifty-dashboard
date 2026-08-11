"""
Verify the GitHub token before trusting it with live state.

Checks, in order: the token is visible to the app; it can READ the alert file
from the repo; and it can WRITE (a no-op round-trip that rewrites the file with
its existing contents, so nothing is lost even if you run it repeatedly).

    python check_repo_store.py            # read-only checks
    python check_repo_store.py --write    # also prove the write path works

A read that works while a write fails means the token is missing the
Contents: Read and write permission — the single most common setup mistake.
"""
import argparse
import sys

from utils import repo_store
from utils.rs_alerts import RS_ALERTS_FILE


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true",
                    help="also perform a harmless write round-trip")
    args = ap.parse_args()

    print("1. token visible to the app ...", end=" ")
    if not repo_store.get_token():
        print("NO")
        print("\n   Nothing is set. Provide one of:")
        print("     export GITHUB_TOKEN=github_pat_...")
        print("     or [github] token = \"...\" in .streamlit/secrets.toml")
        return 1
    tok = repo_store.get_token()
    print(f"yes ({tok[:11]}…{tok[-4:]}, {len(tok)} chars)")

    if repo_store.is_disabled():
        print("   NOTE: REPO_STORE_DISABLED is set — the app will stay local-only.")
        return 1

    print(f"2. target ......................  {repo_store.get_repo()}@{repo_store.get_branch()}")

    print("3. READ  {} ...".format(RS_ALERTS_FILE), end=" ")
    data, sha = repo_store.get_json(RS_ALERTS_FILE, use_cache=False)
    if sha is None and data is None:
        print("could not read")
        print("\n   Either the file does not exist yet (fine — a write creates it),")
        print("   or the token cannot see this repo. Check the token's Repository")
        print("   access includes", repo_store.get_repo())
    else:
        print(f"ok ({len(data) if isinstance(data, list) else '?'} alert(s), sha {str(sha)[:7]})")

    if not args.write:
        print("\nRead checks done. Re-run with --write to verify the write path.")
        return 0

    print("4. WRITE round-trip ...", end=" ")
    payload = data if isinstance(data, list) else []
    ok = repo_store.put_json(RS_ALERTS_FILE, payload,
                             "chore: verify repo_store write access (no-op)")
    if ok:
        print("ok")
        print("\nAll good — RS alerts will sync across devices.")
        return 0
    print("FAILED")
    print("\n   The token can read but not write. Fix: the PAT needs")
    print("   Repository permissions -> Contents: Read and write")
    print("   (a token with only 'Contents: Read' passes step 3 and fails here).")
    return 1


if __name__ == "__main__":
    sys.exit(main())
