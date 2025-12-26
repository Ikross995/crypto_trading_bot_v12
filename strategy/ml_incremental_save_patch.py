"""
🔧 ML INCREMENTAL SAVE PATCH
===========================

Adds incremental saving functionality to AdvancedMLLearningSystem.
Saves ML models every N trades instead of only at shutdown.

Usage:
  from strategy.ml_incremental_save_patch import patch_ml_system
  ml_system = AdvancedMLLearningSystem(config)
  patch_ml_system(ml_system)

"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def patch_ml_system(ml_system):
    """
    Patch ML system to add incremental saving.

    Args:
        ml_system: AdvancedMLLearningSystem instance
    """

    # Add save counter
    ml_system._trades_since_last_save = 0
    ml_system._save_interval = 10  # Save every 10 trades
    ml_system._auto_save_enabled = True

    # Store original warm_start_training
    original_warm_start = ml_system._warm_start_training

    def patched_warm_start_training(self, historical_contexts, historical_outcomes):
        """Patched warm start with auto-save after training."""
        # Call original
        original_warm_start(historical_contexts, historical_outcomes)

        # Save after warm start
        if self._auto_save_enabled:
            logger.info("💾 [WARM_START] Auto-saving models after warm start training...")
            self.save_data()

    # Replace method
    ml_system._warm_start_training = lambda hc, ho: patched_warm_start_training(ml_system, hc, ho)

    # Patch record/train methods if they exist
    if hasattr(ml_system, 'train_on_trade_outcome'):
        original_train = ml_system.train_on_trade_outcome

        async def patched_train_on_trade_outcome(*args, **kwargs):
            """Patched training with incremental save."""
            result = await original_train(*args, **kwargs)

            # Increment counter
            ml_system._trades_since_last_save += 1

            # Auto-save check
            if ml_system._trades_since_last_save >= ml_system._save_interval:
                logger.info(
                    f"💾 [AUTO_SAVE] Saving ML models "
                    f"({ml_system._trades_since_last_save} trades since last save)"
                )
                ml_system.save_data()
                ml_system._trades_since_last_save = 0

            return result

        ml_system.train_on_trade_outcome = patched_train_on_trade_outcome

    logger.info(
        f"✅ [PATCH] ML system patched with incremental save "
        f"(interval: {ml_system._save_interval} trades)"
    )

    return ml_system


def enable_aggressive_save(ml_system, interval: int = 5):
    """
    Enable aggressive saving (for cold start phase).

    Args:
        ml_system: Patched ML system
        interval: Save every N trades
    """
    if hasattr(ml_system, '_save_interval'):
        ml_system._save_interval = interval
        logger.info(f"⚡ [AGGRESSIVE_SAVE] Save interval set to {interval} trades")
    else:
        logger.warning("⚠️ ML system not patched yet!")


def get_save_stats(ml_system) -> dict:
    """Get save statistics."""
    if not hasattr(ml_system, '_trades_since_last_save'):
        return {'patched': False}

    return {
        'patched': True,
        'trades_since_last_save': ml_system._trades_since_last_save,
        'save_interval': ml_system._save_interval,
        'auto_save_enabled': ml_system._auto_save_enabled
    }
