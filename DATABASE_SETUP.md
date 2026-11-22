# RAG-Lines Database Setup Guide

## Current Setup (No External Database Required)

Your RAG-Lines uses **file-based storage** - everything stored locally:

### 1. **Vector Database: ChromaDB** ✅
- **Status:** Built-in, no installation needed
- **Storage:** `./data/chroma_db/` (file-based)
- **How it works:** Stores document embeddings (vector representations)
- **Persistence:** Automatic on disk

### 2. **Application Database: SQLite** ✅
- **Status:** Built-in, no installation needed
- **Storage:** `./data/app.db` (file-based)
- **How it works:** Stores app metadata
- **Persistence:** Automatic on disk

### 3. **Scheduled Jobs: JSON Files** ✅
- **Status:** Built-in, no installation needed
- **Storage:** `./data/scheduled_jobs/*.json` (file-based)
- **How it works:** Persists scheduled job states
- **Persistence:** Automatic on disk

### 4. **Document Storage** ✅
- **Status:** Built-in, no installation needed
- **Storage:** `./data/documents/` (file-based)
- **How it works:** Uploaded documents stored here
- **Persistence:** Automatic on disk

---

## Directory Structure

```
./data/
├── chroma_db/              # Vector database (ChromaDB)
│   ├── index/
│   ├── collections/
│   └── ...
├── app.db                  # SQLite application database
├── documents/              # Ingested documents
└── scheduled_jobs/         # Scheduled job records
    ├── job1.json
    ├── job2.json
    └── ...

./logs/
└── rag-system.log          # Application logs
```

---

## For Production: Database Upgrade Path

When you speak with IT admin **tomorrow**, you can upgrade to enterprise databases:

### Option 1: PostgreSQL (Recommended for Scale)
- For ChromaDB: PgVector extension
- For app data: Native PostgreSQL
- **Setup:** Install PostgreSQL server, update config
- **When to use:** >1M documents, high concurrency

### Option 2: MongoDB
- For scheduled jobs: MongoDB collections
- For document metadata: MongoDB
- **Setup:** Install MongoDB server, update config
- **When to use:** Flexible schema, horizontal scaling

### Option 3: Redis
- For caching query results
- For session management
- **Setup:** Install Redis server, update config
- **When to use:** High-speed caching, leaderboards

---

## Current Configuration

Your `.env` file has these database settings:

```bash
# Database Configuration
DB_TYPE=sqlite                          # Can change to: postgresql, mongodb
DB_CONNECTION_STRING=sqlite:///./data/app.db
VECTORSTORE_TYPE=chroma                 # Vector store type
VECTORSTORE_PATH=./data/chroma_db       # ChromaDB persistent storage
SCHEDULED_JOBS_DIR=./data/scheduled_jobs
DOCUMENT_STORAGE_PATH=./data/documents
```

---

## What You Have NOW (File-Based)

✅ **Advantages:**
- No server setup required
- Works offline
- No additional installation
- Perfect for development/testing
- Easy backup (just copy `./data/` folder)

❌ **Limitations:**
- Single-server only (no distribution)
- Max ~1M documents (performance)
- No multi-user access control
- No enterprise backup features

---

## What IT Admin Can Set Up TOMORROW

Ask IT admin about providing:

### For Production RAG System:

1. **PostgreSQL Database**
   - Server: `postgresql.company.local` or IP
   - Port: 5432 (standard)
   - Credentials: username/password
   - Extensions: pgvector (for embeddings)

2. **Redis Cache (Optional)**
   - Server: `redis.company.local` or IP
   - Port: 6379
   - Password: (if required)

3. **Network Access**
   - Firewall rules for API (port 8000)
   - Firewall rules for LM Studio (port 1235)
   - Internal DNS for database servers

4. **Storage/Backup**
   - Where to store document uploads
   - Backup schedule for databases
   - Disaster recovery procedures

---

## Questions for IT Admin

Print this and give to IT admin:

```
Subject: RAG-Lines Database Setup Request

We're building an AI RAG system that currently uses file-based storage.
For production, we'd need:

1. PostgreSQL database with pgvector extension
   - Purpose: Persistent vector embeddings + app data
   - Estimated size: Start with 100GB
   - Backup requirement: Daily

2. Optional Redis server
   - Purpose: Query result caching
   - Estimated size: 50GB
   - HA/failover: Yes

3. Shared storage mount (for uploads)
   - Access from: Linux server (Ubuntu 20.04+)
   - Capacity: 1TB initially
   - Backup: Yes

Server specs:
- Python 3.11+
- LM Studio or Ollama running
- Ubuntu 20.04+ or CentOS 7+
- 16GB RAM minimum
- 8 CPU cores
- SSD storage

Timeline: Production launch in [your date]
```

---

## Migration Path (When Ready)

When databases are set up:

**Step 1:** Update `.env`
```bash
DB_TYPE=postgresql
DB_CONNECTION_STRING=postgresql://user:pass@db.company.local:5432/rag_db
VECTORSTORE_TYPE=pgvector
```

**Step 2:** Run migration script (I'll create it)
```bash
python scripts/migrate_to_postgres.py
```

**Step 3:** Restart API
```bash
python main.py api
```

Done! Everything continues working, now on enterprise database.

---

## For NOW: Testing

You can immediately test:
1. ✅ Single queries
2. ✅ Batch queries
3. ✅ Document upload/ingestion
4. ✅ Scheduled jobs
5. ✅ All features work locally

Everything is **stored safely** in `./data/` folder.

---

## Backup Your Data (Important!)

Before testing heavily, backup:
```bash
# Copy entire data folder
cp -r ./data/ ./data_backup_$(date +%Y%m%d)
```

Or on Windows:
```bash
xcopy .\data\ .\data_backup_%date:~-4,4%%date:~-10,2%%date:~-7,2%\ /E /I
```

---

## Next Steps

1. **Now:** Test the UI with file-based storage ✅
2. **Tomorrow:** Talk to IT admin, show them this guide
3. **Next week:** Implement PostgreSQL connection
4. **Production:** Deploy with enterprise database

You're all set! Ready to proceed with UI testing? 🚀
