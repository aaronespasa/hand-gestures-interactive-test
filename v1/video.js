// Get the <video> element with id="myVideo"
var video = document.getElementById("myVideo");

var wordsJSON;
var questionsJSON;
var gesturesJSON;

var answers1 = ["1 -> Decreased","2 -> Same","3 -> Increased", "Not Answered"];
var answers2 = ["1 -> 50 years","2 -> 60 years","3 -> 70 years", "Not Answered"];
var answers = [answers1, answers2];

// var startedPoll = false;

$(window).resize(function(){
	updateLayout();
});

String.prototype.capitalize = function() {
    return this.charAt(0).toUpperCase() + this.slice(1);
}

function showPoll() {
	var startTime = parseFloat(questionsJSON[0]["endTime"].replace('s',''));
	return video.currentTime >= startTime;
}

function getVotes(questionEndTime) {
	var one = 0;
	var two = 0;
	var three = 0;

	for (var key in gesturesJSON) {
    if (gesturesJSON.hasOwnProperty(key)) {
			for (j = 0; j < gesturesJSON[key]["one-finger"].length; j++) {
				var value = gesturesJSON[key]["one-finger"][j];
				console.log(value);
				if (value > questionEndTime && value < video.currentTime) {
					one += 1;
				}
			}
			for (j = 0; j < gesturesJSON[key]["two-fingers"].length; j++) {
				var value = gesturesJSON[key]["two-fingers"][j];
				console.log(value);
				if (value > questionEndTime && value < video.currentTime) {
					two += 1;
				}
			}
			for (j = 0; j < gesturesJSON[key]["three-fingers"].length; j++) {
				var value = gesturesJSON[key]["three-fingers"][j];
				console.log(value);
				if (value > questionEndTime && value < video.currentTime) {
					three += 1;
				}
			}
    }
	}

	return [one,two,three,3 - one - two - three];
}

function getPollTitleAndAnswers() {
	var setQuestionStartTime = 0;
	var setQuestionEndTime = 0;
	var answersIndex = 0;
	for (i = 0; i < questionsJSON.length; i++) {
		var questionStartTime = parseFloat(questionsJSON[i]["startTime"].replace('s',''));
		var questionEndTime = parseFloat(questionsJSON[i]["endTime"].replace('s',''));
		var nextIndex = Math.min(i+1, questionsJSON.length - 1);
		var nextQuestionStartTime = parseFloat(questionsJSON[nextIndex]["startTime"].replace('s',''));
		var nextQuestionEndTime = parseFloat(questionsJSON[nextIndex]["endTime"].replace('s',''));
		if (video.currentTime > questionStartTime && video.currentTime < questionEndTime) {
			return ["",["","","",""], [0,0,0,0]];
		} else if (video.currentTime < nextQuestionStartTime) {
			setQuestionStartTime = questionStartTime;
			setQuestionEndTime = questionEndTime;
			break;
		}
		answersIndex = i;
	}

	var questionString = "";

	votes = getVotes(questionEndTime);

	for (i = 0; i < wordsJSON.length; i++) {
		var wordStart = parseFloat(wordsJSON[i]["startTime"].replace('s',''));
		if (wordStart >= questionStartTime && wordStart <= questionEndTime) {
			var wordString = wordsJSON[i]["word"];
			questionString += wordString + " ";
		}
	}
	if (video.currentTime > setQuestionEndTime) {
		questionString = questionString.replace('question','');
		questionString = questionString.replace('give me','');
		questionString = questionString + "?";
		return [questionString, answers[answersIndex], votes];
	}
}

// Assign an ontimeupdate event to the <video> element, and execute a function if the current playback position has changed
video.ontimeupdate = function() {videoTimeUpdated()};

function videoTimeUpdated() {

	transcript = ""
	for (i = 0; i < wordsJSON.length; i++) {
		startTime = parseFloat(wordsJSON[i]["startTime"].replace('s',''));
		if (startTime > video.currentTime - 0.7) {
			break;
		} else {
			transcript += wordsJSON[i]["word"] + " ";
		}
	}
	$("#myText").text(transcript.capitalize());

	updateLayout();
}

function loadJSON(callback, file) {
    var xobj = new XMLHttpRequest();
		xobj.overrideMimeType("application/json");
    xobj.open('GET', file, true);
    xobj.onreadystatechange = function () {
			if (xobj.readyState == 4) {
				// Required use of an anonymous callback as .open will NOT return a value but simply returns undefined in asynchronous mode
				callback(xobj.responseText);
			}
		};
		xobj.send(null);
 }

loadJSON(function(response) {
 // Parse JSON string into object
	 wordsJSON = JSON.parse(response);
}, 'words.json');

loadJSON(function(response) {
 // Parse JSON string into object
	 questionsJSON = JSON.parse(response);
}, 'questions.json');

loadJSON(function(response) {
 // Parse JSON string into object
	 gesturesJSON = JSON.parse(response);
}, 'gestures.json');

function updateLayout() {
	var containerWidth = $('.main').width();
	if (showPoll()) {
		// Video
		$('.video').css('width', containerWidth*0.6);
		// Right
		$('.right').css('width', containerWidth*0.39);
		var rightContainerHeight = $('.right').height();
		// Question
		var questionContainerHeight = $('.question').height();
		var questionMarginTop = questionContainerHeight/2 - 10;
		$('.question').css('margin-top', questionMarginTop);
		$('.question').css('height', rightContainerHeight*0.2 - questionMarginTop);
		// Chart
		var chartHeight = $('#myChart').height();
		var chartContainerHeight = $('.chart').height();
		$('#myChart').css('margin-top',chartContainerHeight/2 - chartHeight/2);
		updateChart();
	} else {
		$('.video').css('width', containerWidth);
		$('.right').css('width', 0);
	}
}

function updateChart() {
	var ctx = document.getElementById('myChart');

	var array = getPollTitleAndAnswers();
	var pollTitle = array[0];
	var pollAnswers = array[1];
	var pollVotes = array[2];

	var myChart = new Chart(ctx, {
	    type: 'bar',
	    data: {
	        labels: [
										pollAnswers[0],
										pollAnswers[1],
										pollAnswers[2],
										pollAnswers[3],
									],
	        datasets: [{
	            label: '# of Votes',
	            data: [
											// Math.floor(Math.random() * 4) + 1,
											// Math.floor(Math.random() * 4) + 1,
											// Math.floor(Math.random() * 4) + 1,
											// Math.floor(Math.random() * 4) + 1
											pollVotes[0],
											pollVotes[1],
											pollVotes[2],
											pollVotes[3]
										],
	            backgroundColor: [
	                'rgba(254, 185, 128, 0.5)',
	                'rgba(71, 107, 179, 0.5)',
									'rgba(220, 115, 255, 0.5)',
	                'rgba(125, 125, 125, 0.5)'
	            ],
	            borderColor: [
	                'rgba(254, 185, 128, 1)',
	                'rgba(71, 107, 179, 1)',
									'rgba(220, 115, 255, 1)',
	                'rgba(125, 125, 125, 1)'
	            ],
	            borderWidth: 1
	        }]
	    },
	    options: {
	        scales: {
	            yAxes: [{
	                ticks: {
	                    beginAtZero: true,
											suggestedMax: 5,
											userCallback: function(label, index, labels) {
                     		if (Math.floor(label) === label) {
                        	return label;
                     		}
                 			}
	                }
	            }]
	        },
					animation: {
        		duration: 0
    			},
					legend: {
    				display: false
					},
					title: {
    				display: true,
						text:"Votes"
					},
					maintainAspectRatio: false,
	    }
	});

	// Set title
	$(".question").text(pollTitle);
}

updateLayout()
