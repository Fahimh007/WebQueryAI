from django.db import models

# Create your models here.

class IndexedSite(models.Model):
    """Tracks websites that have been scraped and indexed into the vector store."""
    url = models.URLField(unique=True)
    title = models.CharField(max_length=500, blank=True)
    chunk_count = models.IntegerField(default=0)
    indexed_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    status = models.CharField(
        max_length=20,
        choices=[('pending', 'Pending'), ('indexed', 'Indexed'), ('failed', 'Failed')],
        default='pending'
    )
    error_message = models.TextField(blank=True)

    def __str__(self):
        return self.url

    class Meta:
        ordering = ['-indexed_at']


class QueryHistory(models.Model):
    """Stores user queries and AI responses for a session."""
    session_key = models.CharField(max_length=40)
    site = models.ForeignKey(IndexedSite, on_delete=models.CASCADE, related_name='queries')
    query = models.TextField()
    answer = models.TextField()
    source_chunks = models.JSONField(default=list)  # Stores relevant source snippets
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"[{self.site.url}] {self.query[:60]}"

    class Meta:
        ordering = ['created_at']