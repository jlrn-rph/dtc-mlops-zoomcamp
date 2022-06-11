from prefect.deployments import DeploymentSpec
from prefect.orion.schemas.schedules import CronSchedule

DeploymentSpec(
    name="cron-schedule-deployment-daniel",
    flow_location="./homework.py",
    schedule=CronSchedule(cron="0 9 15 * *")
)
