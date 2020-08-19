import { Component, OnInit, ViewChild, ElementRef } from "@angular/core";
import { FormGroup, FormControl, Validators } from "@angular/forms";
import { trigger, transition, style, animate } from "@angular/animations";
import { DataService } from "../services/data.service";
import { Router } from "@angular/router";

const fadeAnimation = trigger("fadeAnimation", [
  transition(":enter", [
    style({ opacity: 0 }),
    animate("800ms", style({ opacity: 1 }))
  ]),
  transition(":leave", [
    style({ opacity: 1 }),
    animate("800ms", style({ opacity: 0 }))
  ])
]);

@Component({
  selector: "app-upload",
  templateUrl: "./upload.component.html",
  styleUrls: ["./upload.component.less"],
  animations: [fadeAnimation]
})
export class UploadComponent implements OnInit {
  @ViewChild("labelImport")
  labelImport: ElementRef;

  // DOM control
  loading: boolean = false; // file loading animation
  typeError: boolean = false;
  allowedTypes = [
    "application/msword",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/pdf"
  ];

  // variables
  formImport: FormGroup;
  fileToUpload = [];

  constructor(public dataService: DataService, private router: Router) {
    this.formImport = new FormGroup({
      importFile: new FormControl("", Validators.required)
    });
  }

  ngOnInit() {}

  onFileChange(files: FileList) {
    this.fileToUpload = []; // clear queue
    this.typeError = false; // reset type detection

    this.labelImport.nativeElement.innerHTML = Array.from(files)
      .map(f => f.name)
      .join(", <br />"); // support multiple but not needed

    for (let i = 0; i < files.length; i++) {
      if (this.allowedTypes.includes(files[i].type)) {
        this.fileToUpload.push(files.item(i));
      } else {
        this.typeError = true;
        break;
      }
    }
  }

  confirm() {
    // call API in data service here
    this.loading = true;
    this.dataService.uploadFile(this.fileToUpload).subscribe(
      uploadRes => {
        // map uploadRes
        let docsArr = []
        for (let e in uploadRes) {
          docsArr.push({
            docId: e,
            docName: uploadRes[e]
          });
        }
        this.dataService.setDocs(docsArr);
        setTimeout(() => {
          this.router.navigate(["view-result", this.dataService.docs[0].docId]);
          this.loading = false;
        }, 3000);
      },
      err => {
        console.log("Upload error: ", err);
      }
    );
  }
}
