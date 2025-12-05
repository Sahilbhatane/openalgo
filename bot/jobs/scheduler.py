"""
Job Scheduler
=============
APScheduler-based job scheduling for trading bot.
"""

import logging
from datetime import datetime, time
from dataclasses import dataclass
from typing import Callable, Optional, Dict, Any
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.jobstores.memory import MemoryJobStore
from apscheduler.executors.pool import ThreadPoolExecutor

from ..utils.time_utils import IST, is_trading_day, get_ist_now


logger = logging.getLogger("bot.jobs.scheduler")


@dataclass
class ScheduledJob:
    """Represents a scheduled job"""
    id: str
    name: str
    trigger_time: Optional[time] = None  # For cron jobs
    interval_seconds: Optional[int] = None  # For interval jobs
    func: Optional[Callable] = None
    args: tuple = ()
    kwargs: Dict[str, Any] = None
    trading_days_only: bool = True  # Only run on trading days
    
    def __post_init__(self):
        if self.kwargs is None:
            self.kwargs = {}


class JobScheduler:
    """
    Central scheduler for all bot jobs.
    
    Jobs:
    - Morning Learning (8:00 AM): Analyze market, create trading plan
    - Pre-Market Check (9:05 AM): Verify connections, data availability
    - Market Open (9:15 AM): Start trading
    - Intraday Updates (every 5 min): Check positions, update stops
    - Square Off (3:10 PM): Close all positions
    - EOD Report (3:35 PM): Generate daily report
    - Daily Cleanup (4:00 PM): Clear caches, backup logs
    
    Usage:
        scheduler = JobScheduler()
        scheduler.add_daily_job("morning_learning", time(8, 0), morning_job)
        scheduler.add_daily_job("square_off", time(15, 10), square_off_job)
        scheduler.start()
    """
    
    def __init__(self, timezone=IST):
        """
        Initialize scheduler.
        
        Args:
            timezone: Timezone for scheduling (default: IST)
        """
        self.timezone = timezone
        
        # Configure APScheduler
        jobstores = {
            'default': MemoryJobStore()
        }
        executors = {
            'default': ThreadPoolExecutor(10),
        }
        job_defaults = {
            'coalesce': True,  # Combine missed runs
            'max_instances': 1,  # Only one instance at a time
            'misfire_grace_time': 60,  # 1 minute grace
        }
        
        self._scheduler = BackgroundScheduler(
            jobstores=jobstores,
            executors=executors,
            job_defaults=job_defaults,
            timezone=self.timezone,
        )
        
        self._jobs: Dict[str, ScheduledJob] = {}
        self._running = False
    
    @property
    def is_running(self) -> bool:
        return self._running
    
    def _wrap_job(self, job: ScheduledJob) -> Callable:
        """Wrap job function to add trading day check"""
        def wrapper():
            try:
                # Check if trading day (if required)
                if job.trading_days_only and not is_trading_day():
                    logger.info(f"Skipping job {job.name}: not a trading day")
                    return
                
                logger.info(f"Running job: {job.name}")
                start_time = datetime.now()
                
                # Run the job
                result = job.func(*job.args, **job.kwargs)
                
                elapsed = (datetime.now() - start_time).total_seconds()
                logger.info(f"Job {job.name} completed in {elapsed:.2f}s")
                
                return result
                
            except Exception as e:
                logger.error(f"Job {job.name} failed: {e}", exc_info=True)
                raise
        
        return wrapper
    
    def add_daily_job(
        self,
        job_id: str,
        run_time: time,
        func: Callable,
        name: Optional[str] = None,
        trading_days_only: bool = True,
        args: tuple = (),
        kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Add a job to run once daily at a specific time.
        
        Args:
            job_id: Unique job identifier
            run_time: Time to run (in IST)
            func: Function to execute
            name: Human-readable name
            trading_days_only: Only run on trading days
            args: Positional arguments for func
            kwargs: Keyword arguments for func
        """
        job = ScheduledJob(
            id=job_id,
            name=name or job_id,
            trigger_time=run_time,
            func=func,
            args=args,
            kwargs=kwargs or {},
            trading_days_only=trading_days_only,
        )
        
        self._jobs[job_id] = job
        
        # Create cron trigger
        trigger = CronTrigger(
            hour=run_time.hour,
            minute=run_time.minute,
            timezone=self.timezone,
        )
        
        self._scheduler.add_job(
            self._wrap_job(job),
            trigger,
            id=job_id,
            name=job.name,
            replace_existing=True,
        )
        
        logger.info(f"Added daily job: {job.name} at {run_time}")
    
    def add_interval_job(
        self,
        job_id: str,
        interval_seconds: int,
        func: Callable,
        name: Optional[str] = None,
        trading_days_only: bool = True,
        start_time: Optional[time] = None,
        end_time: Optional[time] = None,
        args: tuple = (),
        kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Add a job to run at regular intervals.
        
        Args:
            job_id: Unique job identifier
            interval_seconds: Seconds between runs
            func: Function to execute
            name: Human-readable name
            trading_days_only: Only run on trading days
            start_time: Don't run before this time
            end_time: Don't run after this time
            args: Positional arguments for func
            kwargs: Keyword arguments for func
        """
        job = ScheduledJob(
            id=job_id,
            name=name or job_id,
            interval_seconds=interval_seconds,
            func=func,
            args=args,
            kwargs=kwargs or {},
            trading_days_only=trading_days_only,
        )
        
        self._jobs[job_id] = job
        
        # Wrap with time window check
        def time_windowed_wrapper():
            now = get_ist_now().time()
            
            if start_time and now < start_time:
                return
            if end_time and now > end_time:
                return
            
            return self._wrap_job(job)()
        
        trigger = IntervalTrigger(seconds=interval_seconds)
        
        self._scheduler.add_job(
            time_windowed_wrapper,
            trigger,
            id=job_id,
            name=job.name,
            replace_existing=True,
        )
        
        logger.info(f"Added interval job: {job.name} every {interval_seconds}s")
    
    def remove_job(self, job_id: str):
        """Remove a scheduled job"""
        if job_id in self._jobs:
            del self._jobs[job_id]
            try:
                self._scheduler.remove_job(job_id)
                logger.info(f"Removed job: {job_id}")
            except Exception:
                pass
    
    def pause_job(self, job_id: str):
        """Pause a scheduled job"""
        try:
            self._scheduler.pause_job(job_id)
            logger.info(f"Paused job: {job_id}")
        except Exception as e:
            logger.error(f"Failed to pause job {job_id}: {e}")
    
    def resume_job(self, job_id: str):
        """Resume a paused job"""
        try:
            self._scheduler.resume_job(job_id)
            logger.info(f"Resumed job: {job_id}")
        except Exception as e:
            logger.error(f"Failed to resume job {job_id}: {e}")
    
    def run_job_now(self, job_id: str):
        """Run a job immediately"""
        if job_id not in self._jobs:
            raise ValueError(f"Unknown job: {job_id}")
        
        job = self._jobs[job_id]
        logger.info(f"Running job immediately: {job.name}")
        
        return self._wrap_job(job)()
    
    def start(self):
        """Start the scheduler"""
        if not self._running:
            self._scheduler.start()
            self._running = True
            logger.info("Scheduler started")
    
    def stop(self, wait: bool = True):
        """Stop the scheduler"""
        if self._running:
            self._scheduler.shutdown(wait=wait)
            self._running = False
            logger.info("Scheduler stopped")
    
    def get_next_run_time(self, job_id: str) -> Optional[datetime]:
        """Get next scheduled run time for a job"""
        try:
            job = self._scheduler.get_job(job_id)
            return job.next_run_time if job else None
        except Exception:
            return None
    
    def get_status(self) -> str:
        """Get scheduler status"""
        status = []
        status.append(f"Scheduler: {'Running' if self._running else 'Stopped'}")
        status.append(f"Jobs: {len(self._jobs)}")
        status.append("")
        
        for job_id, job in self._jobs.items():
            next_run = self.get_next_run_time(job_id)
            next_run_str = next_run.strftime("%H:%M:%S") if next_run else "N/A"
            
            if job.trigger_time:
                schedule = f"Daily at {job.trigger_time.strftime('%H:%M')}"
            elif job.interval_seconds:
                schedule = f"Every {job.interval_seconds}s"
            else:
                schedule = "Unknown"
            
            status.append(f"  {job.name}: {schedule} (next: {next_run_str})")
        
        return "\n".join(status)


def create_trading_scheduler() -> JobScheduler:
    """
    Create a pre-configured scheduler for trading.
    
    Returns a scheduler with standard trading jobs configured.
    The actual job functions need to be set separately.
    """
    scheduler = JobScheduler()
    
    # Note: Job functions should be added by the caller
    # This just sets up the scheduler instance
    
    return scheduler
