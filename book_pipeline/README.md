# AudioBookSum Pipeline

A production-ready ETL pipeline for extracting, processing, and highlighting key points in PDF documents.

## Features

- **Hierarchical Text Extraction**: Extract structured text from PDFs
- **Intelligent Chunking**: Split text into manageable chunks while preserving structure
- **LLM-Powered Analysis**: Identify key insights and interesting points
- **PDF Highlighting**: Generate highlighted PDFs with important sections marked
- **API Interface**: Process documents through a REST API
- **CLI Interface**: Process documents from the command line
- **Performance Monitoring**: Track pipeline metrics with Prometheus
- **Structured Logging**: Comprehensive logging for debugging and monitoring

## Setup and Installation

### Option 1: Using Docker (Recommended)

1. **Prerequisites**:
   - Docker
   - Docker Compose

2. **Quick Start**:
   ```bash
   # Clone the repository
   git clone https://github.com/your-org/audiobook-sum.git
   cd audiobook-sum/book_pipeline

   # Create the required directories
   mkdir -p data output logs cache

   # Create a .env file with your API keys
   echo "GOOGLE_API_KEY=your_google_api_key" > .env

   # Build and start the containers
   docker-compose up -d
   ```

3. **Access the API**: 
   - The API will be available at http://localhost:8000
   - Metrics are available at http://localhost:8001
   - Prometheus dashboard at http://localhost:9090

### Option 2: Local Installation

1. **Prerequisites**:
   - Python 3.10+
   - pip

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   ```bash
   # Create a .env file
   echo "GOOGLE_API_KEY=your_google_api_key" > .env
   ```

4. **Run the API server**:
   ```bash
   python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
   ```

## Usage

### API Usage

1. **Process a PDF file**:
   ```bash
   curl -X POST -F "pdf_file=@/path/to/your/document.pdf" http://localhost:8000/process
   ```

2. **Check the status of a job**:
   ```bash
   curl http://localhost:8000/status/task_12345_document
   ```

3. **List processed files**:
   ```bash
   curl http://localhost:8000/list_files
   ```

4. **Download a processed file**:
   ```bash
   curl -o highlighted.pdf http://localhost:8000/download/document_highlighted.pdf
   ```

### CLI Usage

```bash
# Basic usage
python main.py "/path/to/your/document.pdf"

# Advanced options
python "main.py /path/to/your/document.pdf" --output-dir ./output --cache-dir ./cache --metrics-port 8001 --chunk-size 40000 --chunk-overlap 200 --log-level DEBUG
```

## Monitoring Performance

### Using Prometheus

1. Access the Prometheus dashboard at http://localhost:9090
2. View metrics such as:
   - `pipeline_runs_total`: Total number of pipeline runs
   - `extraction_time_seconds`: Time spent on extraction
   - `transform_time_seconds`: Time spent on transformation
   - `llm_processing_time_seconds`: Time spent on LLM processing
   - `active_pipelines`: Number of active pipeline runs

### Logs

Logs are stored in the `logs` directory and provide detailed information about the pipeline's operation. They contain structured data for easy parsing and analysis.

## Troubleshooting

1. **API Key Issues**:
   - Ensure your Google API key is correctly set in the `.env` file
   - Check permissions for the API key

2. **PDF Processing Errors**:
   - Ensure the PDF is not encrypted or protected
   - For scanned PDFs, try enabling OCR with `--use-mistral`

3. **Performance Issues**:
   - Adjust chunk size with `--chunk-size` for large documents
   - Check the metrics in Prometheus for bottlenecks

## Configuration

Customize the pipeline behavior by modifying these configuration files:

- `src/config.py`: General pipeline settings
- `src/constants.py`: Fixed thresholds and parameters

## License

MIT License