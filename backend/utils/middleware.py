import logging
import time

logger = logging.getLogger("utils.middleware")


class RequestLogMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        start = time.time()
        response = self.get_response(request)
        elapsed = (time.time() - start) * 1000
        if request.path.startswith("/api/"):
            logger.info(
                "%s %s -> %s (%.1fms)",
                request.method,
                request.path,
                response.status_code,
                elapsed,
            )
        return response
