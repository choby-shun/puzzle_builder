# Telegram Puzzle Bot

A Telegram bot that helps users solve jigsaw puzzles by analyzing images and providing step-by-step solutions.

## Project Status

🚧 **Under Development** 🚧

This project is currently under active development. The core functionality is being implemented and tested. Features and APIs may change as development progresses.

## Features

- Image analysis for puzzle piece detection
- Automatic piece matching and placement
- Step-by-step solution guidance
- Support for various puzzle sizes and shapes
- Debug visualization of piece detection and solving process

## Development

### Prerequisites

- Python 3.8+
- OpenCV
- NumPy
- pytest (for testing)

### Running Tests

```bash
# Run all tests
make test

# Run specific test file
pytest tests/test_puzzle_solver.py -v

# Run with debug visualization
make visualize
```

### Project Structure

```
puzzle_builder/
├── bot/
│   ├── puzzles/
│   │   ├── __init__.py
│   │   └── solver.py      # Core puzzle solving logic
│   └── __init__.py
├── tests/
│   └── test_puzzle_solver.py
├── scripts/
│   └── visualize_test.py  # Debug visualization script
├── Makefile
└── README.md
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.