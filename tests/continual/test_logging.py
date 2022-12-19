import logging

from continual.logging import getLogger


def test_log_once(caplog):
    logger = getLogger(__name__, log_once=True)
    with caplog.at_level(logging.WARNING):
        logger.warning("Hey")
        assert caplog.messages == ["Hey"]
        logger.warning("There")
        assert caplog.messages == ["Hey", "There"]
        logger.warning("Hey")
        assert caplog.messages == ["Hey", "There"]  # Hey skipped
