import { Injectable } from "@angular/core";
import { HttpClient } from "@angular/common/http";

import { Observable, throwError } from "rxjs";
import { catchError, retry } from "rxjs/operators";

@Injectable({
  providedIn: "root"
})
export class DataService {
  docs = []; // docs list
  constructor(private http: HttpClient) {}

  setDocs(val) {
    this.docs = val;
    localStorage.setItem('docs', JSON.stringify(val));
  }

  getDocs() {
    if (this.docs.length > 0) {
      return this.docs;
    } else {
      return JSON.parse(localStorage.getItem("docs"));
    }
  }

  getHost() {
    return '/api'; // for deployed
    // return "http://localhost:3002"; // for localhost
  }

  uploadFile(files) {
    const formData = new FormData();
    files.forEach(file => formData.append("file", file));
    return this.http.post(this.getHost() + "/getEula", formData);
  }

  populateResults(docId) {
    const body = { Doc_Id: docId };
    return this.http.post<any>(this.getHost() + "/getResult", body);
  }

  modifyResults(modified) {
    const body = {
      retrain: false,
      modifications: [modified]
    };
    return this.http.post(this.getHost() + "/evaluateClause", body);
  }

  retrainModel() {
    const body = {
      retrain: true,
      modifications: []
    };
    return this.http.post(this.getHost() + "/evaluateClause", body);
  }
}
