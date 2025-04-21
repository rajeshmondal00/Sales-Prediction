from django.db import models

class ForecastResult(models.Model):
    product_category = models.CharField(max_length=100)
    state = models.CharField(max_length=100)
    date = models.DateField()
    predicted_units = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-date']

    def __str__(self):
        return f"{self.product_category} - {self.date}"
