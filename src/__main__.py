import sys

if len(sys.argv) < 2:
    raise ValueError("Expected command: walkforward | query | monthly")

cmd = sys.argv[1]

sys.argv = sys.argv[1:]

if cmd == "walkforward":
    from src.cli.walkforward import main
    main()

elif cmd == "query":
    from src.cli.query import main
    main()

elif cmd == "monthly":
    from src.cli.monthly import main
    main()

else:
    raise ValueError(f"Unknown command: {cmd}")
