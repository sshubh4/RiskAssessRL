"""APScheduler job: ingest → features → validate, daily at 18:00 ET."""
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from pipeline.ingest import ingest
from pipeline.features import compute_features
from pipeline.validate import validate


def pipeline_job(ticker: str = "AAPL") -> None:
    print(f"[scheduler] running pipeline for {ticker}")
    ingest(ticker)
    compute_features(ticker)
    validate(ticker)
    print(f"[scheduler] pipeline complete for {ticker}")


def main() -> None:
    scheduler = BlockingScheduler(timezone="America/New_York")
    scheduler.add_job(
        pipeline_job,
        trigger=CronTrigger(hour=18, minute=0),
        id="daily_pipeline",
        name="Daily AAPL pipeline",
        replace_existing=True,
    )
    print("[scheduler] starting — will run daily at 18:00 ET")
    scheduler.start()


if __name__ == "__main__":
    main()
