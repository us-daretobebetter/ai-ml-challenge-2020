import { NgModule } from "@angular/core";
import { Routes, RouterModule } from "@angular/router";

import { UploadComponent } from "./upload/upload.component";
import { ViewResultComponent } from "./view-result/view-result.component";

const routes: Routes = [
  { path: "", redirectTo: "/upload", pathMatch: "full" },
  { path: "upload", component: UploadComponent },
  { path: "view-result/:docId", component: ViewResultComponent }
];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule {}
