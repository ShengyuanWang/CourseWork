import $ from "jquery";

$(function () {
  $("li:odd").css("background-color", "red");
  $("li:even").css("background-color", "pink");
});
