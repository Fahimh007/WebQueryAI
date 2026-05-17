from django.db import models


class QueryHistory(models.Model):
    url = models.URLField()
    query = models.TextField()
    answer = models.TextField()

    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.query[:50]

    class Meta:
        ordering = ["-created_at"]