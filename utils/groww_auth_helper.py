"""
OpenAlgo Groww Authentication Helper

This script helps validate and configure Groww API credentials.

Usage:
    python -m utils.groww_auth_helper --validate
    python -m utils.groww_auth_helper --generate-test-secret
"""

import argparse
import base64
import os
import sys
from pathlib import Path

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def is_valid_base32(s: str) -> bool:
    """
    Check if a string is valid base32.

    Base32 alphabet: A-Z (uppercase) and 2-7
    Padding with = is allowed at the end.
    """
    # Remove any whitespace and padding
    s = s.strip().rstrip("=").upper()

    # Valid base32 characters
    valid_chars = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ234567")

    for char in s:
        if char not in valid_chars:
            return False

    return len(s) > 0


def validate_groww_credentials() -> dict:
    """
    Validate Groww API credentials from environment.

    Returns:
        dict with validation results
    """
    from dotenv import load_dotenv

    load_dotenv()

    results = {
        "api_key_present": False,
        "api_secret_present": False,
        "api_secret_valid_base32": False,
        "errors": [],
        "suggestions": [],
    }

    api_key = os.getenv("BROKER_API_KEY", "")
    api_secret = os.getenv("BROKER_API_SECRET", "")

    # Check API Key
    if api_key and api_key != "YOUR_BROKER_API_KEY":
        results["api_key_present"] = True

        # Check if it looks like a JWT token
        if api_key.startswith("eyJ"):
            results["suggestions"].append("✅ API key appears to be a valid JWT token")
        else:
            results["suggestions"].append(
                "⚠️ API key doesn't look like a JWT token. Verify with Groww."
            )
    else:
        results["errors"].append(
            "❌ BROKER_API_KEY is not set or is still the placeholder"
        )

    # Check API Secret
    if api_secret and api_secret != "YOUR_BROKER_API_SECRET":
        results["api_secret_present"] = True

        # Check if it's valid base32
        if is_valid_base32(api_secret):
            results["api_secret_valid_base32"] = True
            results["suggestions"].append("✅ API secret is valid base32 format")
        else:
            results["errors"].append(f"❌ BROKER_API_SECRET is NOT valid base32!")
            results["errors"].append(f"   Current value contains invalid characters")
            results["errors"].append(f"   Base32 only allows: A-Z and 2-7")

            # Find invalid characters
            invalid_chars = set()
            for char in api_secret.upper():
                if char not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ234567=":
                    invalid_chars.add(char)

            if invalid_chars:
                results["errors"].append(
                    f"   Invalid characters found: {', '.join(repr(c) for c in invalid_chars)}"
                )

            results["suggestions"].append(
                "\n💡 SOLUTION: Get your correct TOTP secret from Groww:\n"
                "   1. Log in to Groww Trade API portal\n"
                "   2. Go to API credentials section\n"
                "   3. Copy the TOTP Secret (should be base32 encoded)\n"
                "   4. Update BROKER_API_SECRET in your .env file\n"
                "\n   Example of valid base32 secret: JBSWY3DPEHPK3PXP"
            )
    else:
        results["errors"].append(
            "❌ BROKER_API_SECRET is not set or is still the placeholder"
        )

    return results


def generate_test_totp_secret() -> str:
    """Generate a random base32 secret for testing"""
    import secrets

    # Generate 20 random bytes (160 bits) for TOTP
    random_bytes = secrets.token_bytes(20)
    # Encode as base32
    return base64.b32encode(random_bytes).decode("utf-8").rstrip("=")


def test_totp_generation(secret: str) -> bool:
    """Test if TOTP can be generated from the secret"""
    try:
        import pyotp

        totp = pyotp.TOTP(secret)
        code = totp.now()
        print(f"✅ Generated TOTP code: {code}")
        return True
    except Exception as e:
        print(f"❌ Failed to generate TOTP: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Groww Authentication Helper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Validate current credentials
    python -m utils.groww_auth_helper --validate

    # Generate a test secret (for development only)
    python -m utils.groww_auth_helper --generate-test-secret

    # Test TOTP generation with a secret
    python -m utils.groww_auth_helper --test-secret JBSWY3DPEHPK3PXP
        """,
    )

    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate Groww credentials from .env file",
    )

    parser.add_argument(
        "--generate-test-secret",
        action="store_true",
        help="Generate a random test secret (for development only)",
    )

    parser.add_argument(
        "--test-secret", type=str, help="Test TOTP generation with a specific secret"
    )

    args = parser.parse_args()

    if args.validate:
        print("\n" + "=" * 60)
        print("GROWW CREDENTIALS VALIDATION")
        print("=" * 60)

        results = validate_groww_credentials()

        print("\n📋 Status:")
        print(f"   API Key Present: {'✅' if results['api_key_present'] else '❌'}")
        print(
            f"   API Secret Present: {'✅' if results['api_secret_present'] else '❌'}"
        )
        print(
            f"   API Secret Valid Base32: {'✅' if results['api_secret_valid_base32'] else '❌'}"
        )

        if results["errors"]:
            print("\n🚨 Errors:")
            for error in results["errors"]:
                print(f"   {error}")

        if results["suggestions"]:
            print("\n💡 Suggestions:")
            for suggestion in results["suggestions"]:
                print(f"   {suggestion}")

        print("\n" + "=" * 60)

        if results["api_secret_valid_base32"]:
            print("Testing TOTP generation...")
            api_secret = os.getenv("BROKER_API_SECRET", "")
            test_totp_generation(api_secret)

        return 0 if results["api_secret_valid_base32"] else 1

    elif args.generate_test_secret:
        print("\n⚠️  WARNING: This is for TESTING/DEVELOPMENT only!")
        print("    Use your actual Groww TOTP secret for production.\n")

        secret = generate_test_totp_secret()
        print(f"Generated test secret: {secret}")
        print(f"\nAdd to .env file:")
        print(f"BROKER_API_SECRET = '{secret}'")

        print("\nTesting TOTP generation:")
        test_totp_generation(secret)
        return 0

    elif args.test_secret:
        secret = args.test_secret
        print(f"\nTesting secret: {secret}")

        if is_valid_base32(secret):
            print("✅ Secret is valid base32")
            test_totp_generation(secret)
        else:
            print("❌ Secret is NOT valid base32")
            print("   Base32 only allows: A-Z and 2-7")
        return 0

    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
