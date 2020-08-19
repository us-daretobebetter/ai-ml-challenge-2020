import { Component, OnInit, ViewEncapsulation } from "@angular/core";
import {
  trigger,
  transition,
  style,
  animate,
  query,
  stagger
} from "@angular/animations";
import { DataService } from "../services/data.service";
import { NgbModal } from "@ng-bootstrap/ng-bootstrap";
import { MatSnackBar } from "@angular/material/snack-bar";
import { Router, NavigationEnd, ActivatedRoute } from "@angular/router";

const listAnimation = trigger("listAnimation", [
  transition("* <=> *", [
    query(
      ":enter",
      [
        style({ opacity: 0 }),
        stagger("200ms", animate("800ms ease-out", style({ opacity: 1 })))
      ],
      { optional: true }
    ),
    query(":leave", animate("200ms", style({ opacity: 0 })), { optional: true })
  ])
]);

@Component({
  selector: "app-view-result",
  templateUrl: "./view-result.component.html",
  styleUrls: ["./view-result.component.less"],
  encapsulation: ViewEncapsulation.None,
  animations: [listAnimation]
})
export class ViewResultComponent implements OnInit {
  // DOM control

  // Variables
  clauses = [];
  docsList = [];
  activeClause = null;
  modifiedLabel = null;
  selectedDoc = null;
  insightsStat = {};

  constructor(
    public dataService: DataService,
    private modalService: NgbModal,
    private _snackBar: MatSnackBar,
    private router: Router,
    private activatedRoute: ActivatedRoute
  ) {
    router.events.subscribe(event => {
      if (event instanceof NavigationEnd) {
        const urlDocId = activatedRoute.snapshot.params.docId;
        this.docsList = this.dataService.getDocs();
        if (this.docsList.length > 0) {
          for (let i = 0; i < this.docsList.length; i++) {
            if (this.docsList[i].docId === urlDocId) {
              this.fetchClauses(this.docsList[i]);
              break;
            }
          }
        }
      }
    });
  }

  ngOnInit() {

  }

  docNavigate(doc) {
    this.router.navigate(["view-result", doc.docId]);
  }

  fetchClauses(doc) {
    this.selectedDoc = doc;
    this.dataService.populateResults(this.selectedDoc.docId).subscribe(
      res => {
        this.clauses = res;
        this.calcInsights();
      },
      err => {
        console.log("Fetch results error: ", err);
      }
    );
  }

  calcInsights() {
    // this.clauses => this.insightsStat
    try {
      // Number of clauses detected:
      this.insightsStat["NumClauses"] = this.clauses.filter(
        e => e["Predicted_Label"]
      ).length;

      // Number of non-clauses detected:'
      this.insightsStat["NumNonClauses"] = this.clauses.filter(
        e => !e["Predicted_Label"]
      ).length;

      // Number of acceptable clauses:
      this.insightsStat["NumAcceptableClauses"] = this.clauses.filter(
        e => e["Predicted_Label"] == "0.0"
      ).length;

      // Average confidence score of acceptable clauses:
      this.insightsStat["AvgConfAcceptableClauses"] =
        this.clauses.reduce((sum, e) => {
          if (e["Predicted_Label"] == "0.0") {
            sum += e["Prediction_Confidence_Score"];
          }
          return sum;
        }, 0) / this.insightsStat["NumAcceptableClauses"];

      // Number of unacceptable clauses:
      this.insightsStat["NumUnacceptableClauses"] = this.clauses.filter(
        e => e["Predicted_Label"] == "1.0"
      ).length;

      // Average confidence score of unacceptable clauses:
      this.insightsStat["AvgConfUnacceptableClauses"] =
        this.clauses.reduce((sum, e) => {
          if (e["Predicted_Label"] == "1.0") {
            sum += e["Prediction_Confidence_Score"];
          }
          return sum;
        }, 0) / this.insightsStat["NumUnacceptableClauses"];
    } catch (err) {
      console.log("Err calculating insights: ", err);
    }
  }

  back() {
    this.router.navigate(["upload"]);
  }

  print() {
    window.print();
  }

  isClause(clause) {
    return clause.Predicted_Label ? true : false;
  }

  renderPredictionSign(clause) {
    if (clause) {
      // modified label take priority
      // true 0.0 is acceptable
      return clause.Modified_Label !== undefined
        ? clause.Modified_Label === "0.0"
          ? true
          : false
        : clause.Predicted_Label === "0.0"
        ? true
        : false;
    } else return null;
  }

  renderSignalColor(clause) {
    switch (clause.Predicted_Label) {
      case "0.0":
        return {
          color: "rgb(2, 92, 177)",
          shadow: "0 0 10px rgb(46 191 206 / 87%)"
        }; // blue
      case "1.0":
        return {
          color: "rgb(207, 37, 7)",
          shadow: "rgb(207 37 7) 0px 0px 10px"
        }; // orange
      default:
        return { color: "gray", shadow: "0 0 10px gray" };
    }
  }

  isAcceptable() {
    return this.activeClause.Modified_Label
      ? this.activeClause.Modified_Label === "0.0"
      : this.activeClause.Predicted_Label === "0.0";
  }

  openEditModal(content) {
    this.modalService.open(content, { centered: true }).result.then(
      result => {
        // save
        // if modifiedLabel == predictedLabel, clean modified value here and reset to predicted value
        if (this.modifiedLabel === this.activeClause.Predicted_Label) {
          // revert manually corrected label
          delete this.activeClause.Modified_Label;
        } else {
          // record manually corrected label
          this.activeClause.Modified_Label = this.modifiedLabel;
          // update DB records
          this.dataService
            .modifyResults({
              [this.activeClause._id]: this.modifiedLabel === "1.0" ? 1 : 0
            })
            .subscribe(
              res => {
              },
              err => {
                console.log(`Modification error: `, err);
              }
            );
        }
        this.activeClause = null;
      },
      reason => {
        // dismiss
        this.activeClause = null;
      }
    );
  }

  openHelpModal(content) {
    this.modalService.open(content, { centered: true }).result.then(
      result => {
        // save
      },
      reason => {
        // dismiss
      }
    );
  }

  openAnalyticsModal(content) {
    // TODO: render selected docs stat here
    this.modalService.open(content, { centered: true, size: "lg" }).result.then(
      result => {
        // save
      },
      reason => {
        // dismiss
      }
    );
  }

  toggleModifiedLabel(modifiedLabel) {
    this.modifiedLabel = modifiedLabel;
  }

  retrain() {
    this.dataService.retrainModel().subscribe(
      res => {},
      err => {
        console.log("Retrain error: ", err);
      }
    );
    this._snackBar.open(
      "Thanks for your input!  We are re-training the model based on your corrections.",
      "",
      {
        duration: 10000
      }
    );
  }
}
