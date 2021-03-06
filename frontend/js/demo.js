
async function postData(url, data_json={}, pfunction) {
    console.log("input: " + JSON.stringify(data_json))
    fetch(url, {
        method: 'POST',
        cache: 'no-cache',
        headers: {
            'Accept': "application/json, text/plain, */*",
            'Content-Type': "application/json;charset=utf-8"
        },
        //mode: 'no-cors',
        body: JSON.stringify(data_json)
	}).then(resp => resp.json())
		.then(json_output => {pfunction(json_output)}
	);
}

function outputXEL(json) {
	result = document.getElementById("result")
	json_string = JSON.stringify(json)
	console.log(json_string)
	//result.innerHTML +=	json_string.substring(1, json_string.length - 1).replaceAll('\\"', '"')
    if(typeof json.score !== "undefined"){
        if (typeof json.score == "number") {
            result.innerHTML += 'answer: ' + json.answer + '<br>score: '+ json.score.toFixed(2);
        } else { 
            result.innerHTML += 'answer: ' + json.answer + '<br>score: '+ json.score;
        }
     } else{
            //document.getElementById("p").innerHTML = 'The model is loading, please try it again in a few seconds';
	    //result.innerHTML += 'The model is loading, please try it again in a few seconds';
        if(json.error.error.includes('loading')){
	        alert('The model is loading, please try it again in 15-30 seconds!');
            //alert(json.error.error);
            console.log(json.error);
            }
        else{
            alert(json.error.error);
            console.log(json.error);
            }
        }
    //result.innerHTML += 'answer: ' + json.answer + '<br>score: '+ json.score.toFixed(2)
}

function runAnnotation() {
	question = document.getElementById("question").value;
    context = document.getElementById("context").value;
	url_process = "./process";
    data = { "question" : question , "context" : context};

    postData(url_process, data, outputXEL);
    
}

function clearResults(){
	$("#result").html( "" );
}

function formSubmit() {
	clearResults();
	runAnnotation();
	return false;
}
