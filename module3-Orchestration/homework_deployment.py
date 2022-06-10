from prefect.deployments import DeploymentSpec
from prefect.orion.schemas.schedules import CronSchedule
from prefect.flow_runners import SubprocessFlowRunner

DeploymentSpec(
    name="homework_training",
    flow_location="./homework.py",
    schedule=CronSchedule(
        cron="0 9 15 * *"),
    flow_runner=SubprocessFlowRunner(),
    tags=["mlops"]
)