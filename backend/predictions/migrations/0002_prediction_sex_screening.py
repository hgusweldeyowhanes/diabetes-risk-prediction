from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("predictions", "0001_initial"),
    ]

    operations = [
        migrations.AddField(
            model_name="prediction",
            name="sex",
            field=models.CharField(blank=True, max_length=1),
        ),
        migrations.AddField(
            model_name="prediction",
            name="screening",
            field=models.JSONField(blank=True, null=True),
        ),
    ]
