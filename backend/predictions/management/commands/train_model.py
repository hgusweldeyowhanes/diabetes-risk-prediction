from django.core.management.base import BaseCommand

from ml.model_loader import get_predictor, reset_predictor


class Command(BaseCommand):
    help = "Train (or retrain) the advanced diabetes stacking ensemble"

    def add_arguments(self, parser):
        parser.add_argument(
            "--force",
            action="store_true",
            help="Ignore disk cache and retrain from diabetes.csv",
        )

    def handle(self, *args, **options):
        if options["force"]:
            reset_predictor()
        predictor = get_predictor()
        ok = predictor.load_and_train(force_retrain=options["force"])
        if ok:
            auc = predictor.metrics_.get("stack", {}).get("roc_auc")
            self.stdout.write(
                self.style.SUCCESS(
                    "Model ready. stack ROC-AUC=%s  cv_auc=%s" % (auc, predictor.cv_auc_)
                )
            )
        else:
            self.stderr.write(self.style.ERROR("Training failed"))
