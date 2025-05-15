# Kubernetes Model Context Protocol (MCP) Agent

An AI-powered agent that automates Kubernetes YAML file management using Model Context Protocol (MCP). The agent monitors Kubernetes configurations, detects version incompatibilities, and automatically applies necessary changes through the MCP server to maintain project stability during upgrades.

## Features

- Automated Kubernetes YAML file monitoring and analysis
- Version compatibility detection
- Automatic manifest updates via MCP server
- Detailed logging and error handling

## Prerequisites

- Python 3.8 or higher
- Kubernetes cluster access
- MCP server running and accessible

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd kpu_mcp
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment variables:
```bash
cp .env.example .env
```
Edit `.env` file with your MCP server URL and Kubernetes configuration.

## Usage

Run the agent:
```bash
python k8s_agent.py
```

The agent will:
1. Monitor the `manifests/` directory for Kubernetes YAML files
2. Validate version compatibility
3. Automatically update incompatible manifests through the MCP server

## Project Structure

```
├── k8s_agent.py         # Main agent implementation
├── manifests/           # Directory for Kubernetes YAML files
│   └── deployment.yaml  # Sample deployment manifest
├── requirements.txt     # Python dependencies
├── .env.example        # Environment variables template
└── README.md           # Project documentation
```

## Error Handling

The agent uses the `loguru` library for comprehensive logging. Check the console output for:
- Initialization status
- Version compatibility warnings
- Update operations
- Error messages

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request