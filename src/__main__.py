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

elif cmd == "help":
    print(f"""
          Available commands
          - help        : use it to list all the commands     : python -m src help
          - walkforward : run this initially, creates the csv : python -m src walkforward
          - query       : queries the anomalies for date      : python -m src query --date YYYY-MM-DD
          - monthly     : queries market status for the month : python -m src monthly --month YYYY-MM  
    """)

else:
    raise ValueError(f"Unknown command: {cmd}")
