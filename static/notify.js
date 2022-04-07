$(document).ready(function(){
  $("#notify").click(function(){
    this.disable=true;
    $.get("http://127.0.0.1:8889/api/cbf372ac6a78f7207a67ddb667abf076", function(){
      alert("Notification Sent to Police & Authorities");
    });
    this.disable=false;
  });
});
