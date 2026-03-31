# Set up as decault project 
gcloud auth application-default set-quota-project porygon-dataccion

# Enable APIs
gcloud services enable aiplatform.googleapis.com
gcloud services enable modelarmor.googleapis.com
gcloud services enable cloudresourcemanager.googleapis.com

# For RAG
gcloud services enable alloydb.googleapis.com \
                       compute.googleapis.com \
                       cloudresourcemanager.googleapis.com \
                       servicenetworking.googleapis.com \
                       aiplatform.googleapis.com