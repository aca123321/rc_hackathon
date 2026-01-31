from agent import run_test_questions, interactive_mode
from db import get_knowledge_base_stats


def main():
    """Main entry point."""
    import sys

    # Check database state
    stats = get_knowledge_base_stats()
    print(
        f"[DB Check] Total entries: {stats['total_entries']}, With embeddings: {stats['with_embeddings']}, Embedding dim: {stats['embedding_dimension']}"
    )
    if stats["total_entries"] == 0:
        print("[WARNING] Database is empty! Run: python ingestion.py ../novatech-kb")
    elif stats["with_embeddings"] < stats["total_entries"]:
        print(
            f"[WARNING] {stats['total_entries'] - stats['with_embeddings']} entries missing embeddings!"
        )

    if len(sys.argv) > 1:
        if sys.argv[1] == "test":
            # Run test questions
            run_test_questions(
                test_file="../novatech-kb/test_questions.json",
                output_file="test_results.json",
            )
    else:
        # Default to interactive mode
        interactive_mode()


if __name__ == "__main__":
    main()
