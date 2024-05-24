```
# Start prefect server and configure API endpoint
prefect server start
prefect config set PREFECT_API_URL=http://127.0.0.1:4200/api

# Create a work pool of type process (flows run as processes on the local machine)
prefect work-pool create "my-pool" --type process 
prefect work-pool ls

# Deploy a flow by specifying file and entry point, and assign it to a work pool. Then run it by referencing its name.
prefect deploy --name train_flow --pool my-pool orchestration.py:main_flow
prefect deployment run 'main-flow/train_flow'
prefect deployments ls

# Start a worker that monitors a work pool
prefect worker start --pool my-pool

# Deploy a flow by using the interactive menu. This finds available flows in the folder tree, and shows options to pull code from remote storage (such as a git repository). The deployment configuration is saved to the file prefect.yaml. Then run the deployment.
prefect deploy
cat prefect.yaml
prefect deployments ls
prefect deployments run main-flow/deploy-pull-from-remote
```