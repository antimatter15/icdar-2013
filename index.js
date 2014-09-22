var SERVE_URL = 'http://127.0.0.1:1337/'

var fs = require('fs'),
	path = require('path');

var jsfeat = require('./jsfeat-custom')
// 	swt = require('./swt')

var Canvas = require('canvas')
  , Image = Canvas.Image;

var open = require('open');

var BinaryServer = require('binaryjs').BinaryServer;

var http = require('http'),
	url = require('url');

var http = require('http');
var server = http.createServer(function (req, res) {
	res.writeHead(200, {'Content-Type': 'text/html'});
	fs.readFile('console.html', 'utf8', function(err, data){
		res.end(data.replace('__HOST__', url.parse(SERVE_URL).host));	
	})
})

server.listen(
	url.parse(SERVE_URL).port || '80', 
	url.parse(SERVE_URL).hostname
);

var bs = BinaryServer({ server: server })

console.log('Server running at ', SERVE_URL);

var first_client = true;

bs.on('connection', function(client){
	// var file = fs.createReadStream(__dirname + '/flower.png');
	// client.send(file); 
	console.log('client added')
	if(first_client){
		first_client = false;
		do_processing();
	}
});

function broadcast(stream, meta){
	var count = 0;
	for(var id in bs.clients){
		if(bs.clients.hasOwnProperty(id)){
			count++;
			var otherClient = bs.clients[id];
			var send = otherClient.createStream(meta);
			stream.pipe(send);
		}
	}
	if(count == 0){
		// walp no listeners
		stream.on('data', function(){})
	}
}

console.log('waiting for client connection')


var params = {
	// the kernel size for the gaussian blur before canny
	kernel_size: 3,
	// low and high thresh are parameters for the canny edge detector
	low_thresh: 124,
	high_thresh: 204,
	// maximum stroke width, this is the number of iterations
	// the core stroke width transform loop will go through 
	// before giving up and saying that there is no stroke here
	max_stroke: 35,
	// the maximum ratio between adjacent strokes for the 
	// connected components algorithm to consider part of the
	// same actual letter
	stroke_ratio: 2,
	// this is the pixel connectivity required for stuff to happen
	min_connectivity: 4,
	// the minimum number of pixels in a connected component to
	// be considered a candidate for an actual letter
	min_area: 30, //default: 38
	// maximum stroke width variation allowed within a letter
	std_ratio: 0.83,
	// maximum aspect ratio to still be considered a letter
	// for instance, a really long line wouldn't be considered
	// a letter (obviously if this number is too low, it'll start
	// excluding l's 1's and i's which would be bad)
	aspect_ratio: 10, // default: 8
	// maximum ratio between the median thicknesses of adjacent 
	// letters to be considered part of the same line
	thickness_ratio: 3,
	// maximum ratio between adjacent letter heights to be considered
	// part of the same line
	height_ratio: 2.5, // original: 1.7
	
	// for some reason it's much more effective with non-integer scales
	scale: 1.3,
	// scale: 1.8,

	// text_angle: Math.PI / 6
	letter_occlude_thresh: 7, //default 3
	
	// otsu parameter for word breakage
	breakdown_ratio: 0.4,
	// something something making lines
	elongate_ratio: 1.9,
	// maximum number of surrounding pixels to explore during the
	// flood fill swt augmentation stage
	max_substroke: 15,
	// linespacing things for spacing lines, used in forming paragraphs/regions
	min_linespacing: 0.1, // this is for overlap
	max_linespacing: 1.7,
	// otsu breakdown ratio for splitting a paragraph
	col_breakdown: 0.3,
	// the maximum fraction of the width of the larger of two adjacent lines 
	// by which an alignment may be offset in one column
	max_misalign: 0.1,
	// the first one is frac of the smaller area, the second is frac of bigger area
	col_mergethresh: 0.3,

	lettersize: 0.4, // maximum difference between median letter weights of adjacent lines
	// letter weight is defined as the number of pixels per letter divided by the width
	// which is because quite often entire words get stuck together as one letter
	// and medians are used because it's a more robust statistic, this actually works
	// remarkably well as a metric

	// debugs!?!?
	debug: false,
	
	chunk_size: 250,
	chunk_overlap: 90,

}

console.groupCollapsed = function(){}
console.groupEnd = function(){}

// maybe in the future we should replace this with a strongly
// connected components algorithm (or have some spatial heuristic to
// determine how wise it would be to consider the connection valid)
function connected_swt(swt, params){
	var dx8 = [-1, 1, -1, 0, 1, -1, 0, 1];
	var dy8 = [0, 0, -1, -1, -1, 1, 1, 1];
	var width = swt.cols, 
		height = swt.rows;

	var marker = new jsfeat.matrix_t(width, height, jsfeat.U8C1_t)
	var contours = []
	
	for(var i = 0; i < width * height; i++){
		if(marker.data[i] || !swt.data[i]) continue;

		var ix = i % width, iy = Math.floor(i / width)
		
		marker.data[i] = 1
		var contour = []
		var stack = [i]
		var closed;
		
		while(closed = stack.shift()){
			contour.push(closed)
			var cx = closed % width, cy = Math.floor(closed / width);
			var w = swt.data[closed];
			for(var k = 0; k < 8; k++){
				var nx = cx + dx8[k]
				var ny = cy + dy8[k]
				var n = ny * width + nx;

				if(nx >= 0 && nx < width &&
				   ny >= 0 && ny < height &&
				   swt.data[n] &&
				   !marker.data[n] &&
				   swt.data[n] <= params.stroke_ratio * w &&
				   swt.data[n] * params.stroke_ratio >= w){
					marker.data[n] = 1
					// update the average stroke width
					w = (w * stack.length + swt.data[n]) / (stack.length + 1)
					stack.push(n)
				}
			}
		}
		// contours.push(contour)
		if(contour.length >= params.min_area){
			contours.push(contour)	
		}
	}
	return contours
}





function connected_priority(masked){
	var dx8 = [-1, 1, -1, 0, 1, -1, 0, 1];
	var dy8 = [0, 0, -1, -1, -1, 1, 1, 1];
	var width = masked.cols, 
		height = masked.rows;

	var marker = new jsfeat.matrix_t(width, height, jsfeat.U8C1_t)
	var contours = []
	
	var min_area = 10;

	var big_queue = new HeapQueue(function(b, a){
		return masked.data[a] - masked.data[b]
	})
	for(var i = 0; i < width * height; i++){
		if(!masked.data[i]) continue;
		big_queue.push(i)
	}

	// for(var i = 0; i < width * height; i++){
	
	while(big_queue.length){
		var i = big_queue.pop()
		if(marker.data[i] || !masked.data[i]) continue;

		var ix = i % width, iy = Math.floor(i / width)
		
		marker.data[i] = 1
		var contour = []
		// var stack = [i]
		var stack = new HeapQueue(function(b, a){
			return masked.data[a] - masked.data[b]
		})
		stack.push(i)
		var w = masked.data[i];
		var counter = 0, mean = 0, M2 = 0;

		while(stack.length){
			var closed = stack.pop()

			contour.push(closed)
			var cx = closed % width, cy = Math.floor(closed / width);
			
			counter++
			var delta = masked.data[closed] - mean
			mean = mean + delta / counter
			M2 += delta * (masked.data[closed] - mean)

			for(var k = 0; k < 8; k++){
				var nx = cx + dx8[k]
				var ny = cy + dy8[k]
				var n = ny * width + nx;

				var std = Math.sqrt(M2/(counter - 1));

				if(nx >= 0 && nx < width &&
				   ny >= 0 && ny < height &&
				   masked.data[n] &&
				   !marker.data[n]
				   ){
					marker.data[n] = 1
					// console.log(marker.data[n] - w)
				    if(//(
				    	// (masked.data[n] <= max_ratio * w &&
				   		// masked.data[n] * max_ratio >= w) || 
				   		// Math.abs(masked.data[n] - w) < max_diff ||
				   		// masked.data[n] > 80
				   		// w - masked.data[n] < 10
				   		// masked.data[n] > 0
				   		// masked.data[n] >= Math.pow(w, 0.8)
				   		// masked.data[n] > 0
				   		// this is, for the record, an empirically
				   		// derived magic number, and by empirically
				   		// derived, i mean it was pulled out of
				   		// my ass, because it really isn't anything
				   		// of that sort
				   		Math.pow(masked.data[n], 1.5) > w
				   		// masked.data[n] / w > 0.5
				   		// w - masked.data[n] < 50
				   		// masked.data[n] > mean - 8 * std
				   		){
				    	// update the average stroke width
						w = (w * stack.length + masked.data[n]) / (stack.length + 1)


						stack.push(n)
				    }else{
						contour.push(n)
				    }
				}
			}
		}

		// contours.push(contour)
		if(contour.length >= min_area){
			contours.push(contour)	
		}
	}
	// visualize_matrix(marker)
	return contours
}


// This code is tracked by and can be found on 
// https://github.com/naptha/rgb-lab

// it's derived from some of the following:
// https://github.com/harthur/color-convert
// https://github.com/THEjoezack/ColorMine/blob/master/ColorMine/ColorSpaces/Comparisons/Cie94Comparison.cs

function deltaE(labA, labB){
  var deltaL = labA[0] - labB[0];
  var deltaA = labA[1] - labB[1];
  var deltaB = labA[2] - labB[2];
  var c1 = Math.sqrt(labA[1] * labA[1] + labA[2] * labA[2]);
  var c2 = Math.sqrt(labB[1] * labB[1] + labB[2] * labB[2]);
  var deltaC = c1 - c2;

  var deltaH = deltaA * deltaA + deltaB * deltaB - deltaC * deltaC;
  deltaH = deltaH < 0 ? 0 : Math.sqrt(deltaH);

  var sc = 1.0 + 0.045 * c1;
  var sh = 1.0 + 0.015 * c1;

  var deltaLKlsl = deltaL / (1.0);
  var deltaCkcsc = deltaC / (sc);
  var deltaHkhsh = deltaH / (sh);
  var i = deltaLKlsl * deltaLKlsl + deltaCkcsc * deltaCkcsc + deltaHkhsh * deltaHkhsh;
  return i < 0 ? 0 : Math.sqrt(i);
}

function lab2rgb(lab){
  var y = (lab[0] + 16) / 116,
      x = lab[1] / 500 + y,
      z = y - lab[2] / 200,
      r, g, b;

  x = 0.95047 * ((x * x * x > 0.008856) ? x * x * x : (x - 16/116) / 7.787);
  y = 1.00000 * ((y * y * y > 0.008856) ? y * y * y : (y - 16/116) / 7.787);
  z = 1.08883 * ((z * z * z > 0.008856) ? z * z * z : (z - 16/116) / 7.787);

  r = x *  3.2406 + y * -1.5372 + z * -0.4986;
  g = x * -0.9689 + y *  1.8758 + z *  0.0415;
  b = x *  0.0557 + y * -0.2040 + z *  1.0570;

  r = (r > 0.0031308) ? (1.055 * Math.pow(r, 1/2.4) - 0.055) : 12.92 * r;
  g = (g > 0.0031308) ? (1.055 * Math.pow(g, 1/2.4) - 0.055) : 12.92 * g;
  b = (b > 0.0031308) ? (1.055 * Math.pow(b, 1/2.4) - 0.055) : 12.92 * b;

  return [Math.max(0, Math.min(1, r)) * 255, 
          Math.max(0, Math.min(1, g)) * 255, 
          Math.max(0, Math.min(1, b)) * 255]
}


function rgb2lab(rgb){
  var r = rgb[0] / 255,
      g = rgb[1] / 255,
      b = rgb[2] / 255,
      x, y, z;

  r = (r > 0.04045) ? Math.pow((r + 0.055) / 1.055, 2.4) : r / 12.92;
  g = (g > 0.04045) ? Math.pow((g + 0.055) / 1.055, 2.4) : g / 12.92;
  b = (b > 0.04045) ? Math.pow((b + 0.055) / 1.055, 2.4) : b / 12.92;

  x = (r * 0.4124 + g * 0.3576 + b * 0.1805) / 0.95047;
  y = (r * 0.2126 + g * 0.7152 + b * 0.0722) / 1.00000;
  z = (r * 0.0193 + g * 0.1192 + b * 0.9505) / 1.08883;

  x = (x > 0.008856) ? Math.pow(x, 1/3) : (7.787 * x) + 16/116;
  y = (y > 0.008856) ? Math.pow(y, 1/3) : (7.787 * y) + 16/116;
  z = (z > 0.008856) ? Math.pow(z, 1/3) : (7.787 * z) + 16/116;

  return [(116 * y) - 16, 500 * (x - y), 200 * (y - z)]
}


/*
  This implementation is very loosely based off js-priority-queue
  by Adam Hooper from https://github.com/adamhooper/js-priority-queue
  
  The js-priority-queue implementation seemed a teensy bit bloated
  with its require.js dependency and multiple storage strategies
  when all but one were strongly discouraged. So here is a kind of 
  condensed version of the functionality with only the features that
  I particularly needed. 

  Using it is pretty simple, you just create an instance of HeapQueue
  while optionally specifying a comparator as the argument:

  var heapq = new HeapQueue()

  var customq = new HeapQueue(function(a, b){
  	// if b > a, return negative
  	// means that it spits out the smallest item first
	return a - b
  })

  Note that in this case, the default comparator is identical to
  the comparator which is used explicitly in the second queue.

  Once you've initialized the heapqueue, you can plop some new
  elements into the queue with the push method (vaguely reminiscent
  of typical javascript arays)

  heapq.push(42);
  heapq.push("kitten");

  The push method returns the new number of elements of the queue.

  You can push anything you'd like onto the queue, so long as your
  comparator function is capable of handling it. The default 
  comparator is really stupid so it won't be able to handle anything
  other than an number by default.

  You can preview the smallest item by using peek.

  heapq.push(-9999)
  heapq.peek() ==> -9999

  The useful complement to to the push method is the pop method, 
  which returns the smallest item and then removes it from the
  queue.

  heapq.push(1)
  heapq.push(2)
  heapq.push(3)
  heapq.pop() ==> 1
  heapq.pop() ==> 2
  heapq.pop() ==> 3
*/

function HeapQueue(cmp){
	this.cmp = (cmp || function(a, b){ return a - b });
	this.length = 0;
	this.data = []
}

HeapQueue.prototype.peek = function(){
	return this.data[0]
}
HeapQueue.prototype.push = function(value){
	this.data.push(value);
	var pos = this.data.length - 1,
		parent, x;
	while(pos > 0){
		parent = (pos - 1) >>> 1;
		if(this.cmp(this.data[pos], this.data[parent]) < 0){
			x = this.data[parent]
			this.data[parent] = this.data[pos];
			this.data[pos] = x;
			pos = parent;
		}else break;
	}
	return ++this.length;
}
HeapQueue.prototype.pop = function(){
	var ret = this.data[0],
		last_val = this.data.pop();
	this.length--;
	if(this.data.length > 0){
		this.data[0] = last_val;
		var pos = 0,
			last = this.data.length - 1,
			left, right, minIndex, x;
		while(1){
			left = (pos << 1) + 1;
			right = left + 1;
			minIndex = pos;
			if(left <= last && this.cmp(this.data[left], this.data[minIndex]) < 0) minIndex = left;
			if(right <= last && this.cmp(this.data[right], this.data[minIndex]) < 0) minIndex = right;
			if(minIndex !== pos){
				x = this.data[minIndex]
				this.data[minIndex] = this.data[pos]
				this.data[pos] = x;
				pos = minIndex
			}else break;
		}
	}
	return ret
}


// this is the typical union-find algorithm derived from

// https://github.com/mikolalysenko/union-find/blob/master/index.js

function UnionFind(count) {
	this.roots = new Array(count);
	this.ranks = new Array(count);
	
	for(var i=0; i<count; ++i) {
		this.roots[i] = i;
		this.ranks[i] = 0;
	}
}

UnionFind.prototype.length = function() {
	return this.roots.length;
}

UnionFind.prototype.makeSet = function() {
	var n = this.roots.length;
	this.roots.push(n);
	this.ranks.push(0);
	return n;
}

UnionFind.prototype.find = function(x) {
	var roots = this.roots;
	while(roots[x] !== x) {
		var y = roots[x];
		roots[x] = roots[y];
		x = y;
	}
	return x;
}

UnionFind.prototype.link = function(x, y) {
	var xr = this.find(x)
		, yr = this.find(y);
	if(xr === yr) {
		return;
	}
	var ranks = this.ranks
		, roots = this.roots
		, xd    = ranks[xr]
		, yd    = ranks[yr];
	if(xd < yd) {
		roots[xr] = yr;
	} else if(yd < xd) {
		roots[yr] = xr;
	} else {
		roots[yr] = xr;
		++ranks[xr];
	}
}


// this is a port of something from libccv which
// is a port of something from opencv which is 
// a port of an algorithm in some textbook from
// somewhere

// it has rough functional parity with ccv_array_group
// and cvSeqPartition and the union-find algorithm
// except rather than returning a list of list 
// indicies in case one is so inclined to construct
// a list, it actually just returns the list

// this is a quadratic algorithm as far as I'm aware
// which means that the is_equal function will be called
// n(n - 1) times where n is the length of your elements
// array. For things with large numbers of elements,
// this can become very slow.

// it might be wise because of this to inform the
// algorithm with some notion of geometry. i.e.
// "these elements are really really far apart
// so they probably don't have anything to do with
// each other so lets just kind of let them do
// their thing and have incestuous relations with
// people closer to them"

function equivalence_classes(elements, is_equal){
	var node = []
	for(var i = 0; i < elements.length; i++){
		node.push({
			parent: 0,
			element: elements[i],
			rank: 0
		})
	}
	for(var i = 0; i < node.length; i++){
		var root = node[i]
		while(root.parent){
			root = root.parent;
		}
		for(var j = 0; j < node.length; j++){
			if(i == j) continue;
			if(!is_equal(node[i].element, node[j].element)) continue;
			var root2 = node[j];
			while(root2.parent){
				root2 = root2.parent;
			}
			if(root2 != root){
				if(root.rank > root2.rank){
					root2.parent = root;
				}else{
					root.parent = root2;
					if(root.rank == root2.rank){
						root2.rank++  
					}
					root = root2;
				}
				var node2 = node[j];
				while(node2.parent){
					var temp = node2;
					node2 = node2.parent;
					temp.parent = root;
				}
				var node2 = node[i];
				while(node2.parent){
					var temp = node2;
					node2 = node2.parent;
					temp.parent = root;
				}
			}
		}
	}
	var index = 0;
	var clusters = [];
	for(var i = 0; i < node.length; i++){
		var j = -1;
		var node1 = node[i]
		while(node1.parent){
			node1 = node1.parent
		}
		if(node1.rank >= 0){
			node1.rank = ~index++;
		}
		j = ~node1.rank;

		if(clusters[j]){
			clusters[j].push(elements[i])
		}else{
			clusters[j] = [elements[i]]
		}
	}
	return clusters;
}

// HIC SUNT DRACONES

function partial_swt(src, params){
	var width = src.width, height = src.height;
	
	var img_u8 = new jsfeat.matrix_t(width, height, jsfeat.U8C1_t)
	var img_dxdy = new jsfeat.matrix_t(width, height, jsfeat.S32C2_t);

	console.time("image processing")
	jsfeat.imgproc.grayscale(src.data, img_u8.data)
	// visualize_matrix(img_u8)
	jsfeat.imgproc.sobel_derivatives(img_u8, img_dxdy)
	jsfeat.imgproc.gaussian_blur(img_u8, img_u8, params.kernel_size, 0)
	jsfeat.imgproc.canny(img_u8, img_u8, params.low_thresh, params.high_thresh)

	console.timeEnd("image processing")

	function extract_regions(direction){
		console.groupCollapsed(direction == 1 ? 'light on dark' : 'dark on light')
		console.time('total region extraction')
		params.direction = direction;
		var result = raw_swt(img_u8, img_dxdy, params);
		var swt = result.swt, strokes = {};


		function wrap_contours(points){
			var size = points.length;
			var x0 = Infinity, y0 = Infinity, x1 = 0, y1 = 0;
			var m10 = 0, m01 = 0, m11 = 0, m20 = 0, m02 = 0;
			var swtsum = 0, swtvar = 0, swts = [];
			var marksum = 0;
			var y_coords = []

			for(var i = 0; i < size; i++){
				var p = points[i];
				var x = p % width, y = Math.floor(p / width);
				x0 = Math.min(x0, x); y0 = Math.min(y0, y);
				x1 = Math.max(x1, x); y1 = Math.max(y1, y);

				y_coords.push(y)

				m10 += x; m01 += y;
				m11 += x * y;
				m20 += x * x; m02 += y * y;
				swtsum += swt.data[p];
				
				if(marker) marksum += marker.data[p];

				swts.push(swt.data[p]);
			}

			var aspect_ratio = Math.max(x1 - x0, y1 - y0) / Math.min(x1 - x0, y1 - y0)
			
			var mean = swtsum / size;
			
			for(var i = 0; i < size; i++){
				var p = points[i];
				swtvar += (swt.data[p] - mean) * (swt.data[p] - mean)
			}
			var xc = m10 / size, yc = m01 / size;
			var af = m20 / size - xc * xc;
			var bf = 2 * (m11 / size - xc * yc)
			var cf = m02 / size - yc * yc;
			var delta = Math.sqrt(bf * bf + (af - cf) * (af - cf));
			var ratio = Math.sqrt((af + cf + delta) / (af + cf - delta));
			ratio = Math.max(ratio, 1 / ratio)

			// if(ratio > params.aspect_ratio && !is_L) return;
			if(ratio > params.aspect_ratio) return;

			var median = swts.sort(function(a, b){return a - b})[Math.floor(swts.length / 2)]
			var std = Math.sqrt(swtvar / size);
			// if(std > mean * params.std_ratio) return;
			var area = (x1 - x0 + 1) * (y1 - y0 + 1)
			
			if(size / area < 0.1) return;


			var cy = y0 + (y1 - y0) / 2,
				cx = x0 + (x1 - x0) / 2;

			// if(x0 == 0 || y0 == 0 || y1 == height - 1 || x1 == width - 1) return;

			// x-axis border touching is okay because we dont define our 
			// clipping boundaries by them (that doesnt really make sense
			// but im about to go to bed) so like our little chunks are all
			// full width so things touching the edge arent artifacts

			if(y0 == 0 || y1 == height - 1) return;
			
			return {
				x0: x0,
				y0: y0,
				y1: y1,
				x1: x1,
				cx: cx,
				cy: cy,
				width: x1 - x0 + 1,
				height: y1 - y0 + 1,
				size: size,
				// color: dominant_color(colors, direction),
				// color: [sr/size, sg/size, sb/size],
				// color: domcolor,
				ratio: (x1 - x0) / (y1 - y0), 
				ratio2: ratio,
				std: std,
				mean: mean,
				medy: y_coords.sort(function(a, b){ return a - b })[Math.floor(y_coords.length / 2)] - cy,
				area: area,
				contours: points,
				markweight: marksum / size,
				thickness: median
			}
		}


		function wrap_lines(letters){
			if(letters.length == 0) return null;

			letters = letters.sort(function(a, b){ return a.cx - b.cx })

			// var sumx = 0, sumx2 = 0, sumxy = 0, sumy = 0, sumy2 = 0;
			var size = 0;

			var x0 = Infinity, y0 = Infinity, x1 = 0, y1 = 0, hs = 0;
			for(var i = 0; i < letters.length; i++){
				var letter = letters[i];
				x0 = Math.min(x0, letter.x0); y0 = Math.min(y0, letter.y0);
				x1 = Math.max(x1, letter.x1); y1 = Math.max(y1, letter.y1);
				size += letter.size
				hs += letter.height
				// sumx += letter.cx; sumy += letter.cy;
				// sumx2 += letter.cx * letter.cx; sumy2 += letter.cy * letter.cy;
				// sumxy += letter.cx * letter.cy;
			}

			// var n = letters.length;
			// var dy = (n * sumxy - sumx * sumy);
			// var dx = (n * sumx2 - sumx * sumx);
			// var yi = (sumy * sumx2 - sumx * sumxy) / (n * sumx2 - sumx * sumx);
			// var r = (sumxy - sumx * sumy / n) / Math.sqrt((sumx2 - sumx*sumx/n) * (sumy2 - sumy*sumy/n));


			var slopes = []
			// This is an implementation of a Theil-Sen estimator
			// it's like actually really simple, it's just the median
			// of the slopes between every existing pair of points
			for(var i = 0; i < letters.length; i++){
				var li = letters[i];
				for(var j = 0; j < i; j++){
					var lj = letters[j];
					slopes.push((li.cy - lj.cy) / (li.cx - lj.cx))
				}
			}
			var dydx = slopes.sort(function(a, b){ return a - b })[Math.floor(slopes.length/2)]


			var cx = x0 / 2 + x1 / 2,
				cy = y0 / 2 + y1 / 2;

			var yr0 = Infinity, yr1 = -Infinity, sh = 0, st = 0;
			for(var i = 0; i < letters.length; i++){
				var letter = letters[i];
				var y_pred = (letter.cx - cx) * dydx + cy
				yr0 = Math.min(yr0, letter.y0 - y_pred)
				yr1 = Math.max(yr1, letter.y1 - y_pred)
				sh += letter.height
				st += letter.thickness
			}

			var lettersize = letters.map(function(e){
				return e.size / e.width
			}).sort(function(a, b){return a - b})[Math.floor(letters.length / 2)]



			// console.log('letter '+lettersize)

			// approximate the x-height of some line of text
			// as the height of the smallest character whose
			// height is larger than half the average character
			// height
			var xheight = letters.map(function(e){
				return e.height
			}).filter(function(e){
				// return e > (yr1 - yr0) / 3
				return e <= (hs / letters.length)
			}).sort(function(a, b){
				return a - b
			}).slice(-1)[0]

			// var weight = letters.map(function(e){
			// 	return e.thickness
			// }).sort(function(a, b){return a - b})[Math.floor(letters.length / 2)] / xheight

			// var weights = letters.filter(function(e){
			// 	return e.height <= (hs / letters.length)
			// }).map(function(e){
			// 	return e.size / e.width / e.height
			// }).sort(function(a, b){return a - b});

			// var weight = weights[Math.floor(weights.length / 2)]

			return {
				letters: letters,
				lettercount: letters.length,
				lettersize: lettersize,
				// weight: weight,
				size: size,
				lineheight: yr1 - yr0,
				xheight: xheight,
				avgheight: sh / letters.length,
				direction: direction,
				// angle: Math.atan2(dy, dx),
				angle: Math.atan(dydx),
				thickness: st / letters.length,
				// r2: r * r,
				x0: x0,
				y0: y0,
				y1: y1,
				x1: x1,
				cx: cx,
				cy: cy,
				width: x1 - x0 + 1,
				height: y1 - y0 + 1,
				area: (x1 - x0) * (y1 - y0)
			}
		}


		console.time('connected swt')
		var contours = connected_swt(swt, params)
		console.timeEnd('connected swt')

		params.marker = swt

		console.time("wrap contours 1")
		var letters = contours
			.map(wrap_contours)
			.filter(function(e){
				if(!e) return false;
				if(e.std > e.mean * params.std_ratio) return false;
				// if(e.size < 10) return false;
				return true
			})
		console.timeEnd("wrap contours 1")

		letters = exclude_occlusions(letters, width, height, params)

		if(params.debug) visualize_matrix(swt, letters);
		// letters = letters.filter(function(a){
		// 	return !letters.some(function(b){
		// 		var width = Math.min(a.x1, b.x1) - Math.max(a.x0, b.x0),
		// 			height = Math.min(a.y1, b.y1) - Math.max(a.y0, b.y0);
		// 		return width > 0 && height > 0
		// 	})
		// })


		// letters = exclude_occlusions(letters, width, height, params)

		var lines = find_lines(letters, params)
			.filter(function(e){return e.length > 1})
			.map(wrap_lines)
			.filter(function(e){
				return e.lettercount > 3 || (e.lettercount > 2 && Math.abs(e.angle) < 0.1)
			});
		
		if(params.debug) visualize_matrix(swt, lines);

		// params.marker = swt
		if(typeof window == 'object' && params.debug){
			blargh = visualize_matrix(swt, letters);
			lines.forEach(function(line){
				blargh.beginPath()
				var colors = ['green', 'blue', 'red', 'purple', 'orange', 'yellow']
				blargh.strokeStyle = colors[Math.floor(colors.length * Math.random())]
				blargh.lineWidth = 3

				line.letters
					.sort(function(a, b){ return a.cx - b.cx })
					.forEach(function(letter){
						blargh.lineTo(letter.cx, letter.cy)
					})

				blargh.stroke()
			})
		}

		var marker = filter_regions(lines, src)
		
		console.time("flood contours")
		var letters = connected_priority(marker, params)
			.map(wrap_contours)
			.filter(function(e){
				return e
			})
		console.timeEnd("flood contours")
		
		letters = exclude_occlusions(letters, width, height, params)


		

		if(typeof window == 'object' && params.debug){
			blargh = visualize_matrix(marker, letters);
			// lines.forEach(function(line){
			// 	blargh.beginPath()
			// 	var colors = ['green', 'blue', 'red', 'purple', 'orange', 'yellow']
			// 	blargh.strokeStyle = colors[Math.floor(colors.length * Math.random())]
			// 	blargh.lineWidth = 3

			// 	line.letters
			// 		.sort(function(a, b){ return a.cx - b.cx })
			// 		.forEach(function(letter){
			// 			blargh.lineTo(letter.cx, letter.cy)
			// 		})

			// 	blargh.stroke()
			// })
		}
		
		var lines = find_lines(letters, params)
			.filter(function(e){return e.length > 1})
			.map(wrap_lines)



		// console.log(lines)
		// merge the adjacent lines
		for(var i = 0; i < 2; i++){ // do it twice because sometimes it misses something on the first go
			lines = equivalence_classes(lines, function(r_bb, l_bb){
				var y_overlap = Math.min(r_bb.y1, l_bb.y1) - Math.max(r_bb.y0, l_bb.y0);
				if(y_overlap <= 0) return false;
				var frac_overlap = y_overlap / Math.min(r_bb.height, l_bb.height)
				if(frac_overlap < 0.8) return false;
				var x_dist = Math.max(r_bb.x0, l_bb.x0) - Math.min(r_bb.x1, l_bb.x1)
				if(x_dist < 0) return false;
				if(x_dist > 0.2 * Math.max(r_bb.width, l_bb.width)) return false;

				if(x_dist > 3 * Math.max(r_bb.height, l_bb.height)) return false;

				var max_ang = 0.2; // this merger breaks down with too much angle
				if(Math.abs(r_bb.angle) > max_ang || Math.abs(r_bb.angle) > max_ang) return false;
				
				if(Math.max(r_bb.height, l_bb.height) / Math.min(r_bb.height, l_bb.height) > 1.4) return false;

				// if(Math.abs(r_bb.lettersize - l_bb.lettersize) / Math.min(r_bb.lettersize, l_bb.lettersize) > params.lettersize) return false;

				return true
			}).map(function(cluster){
				if(cluster.length == 1) return cluster[0];
				return wrap_lines([].concat.apply([], cluster.map(function(e){return e.letters})))
			})
		}
		


		console.time("split lines")
		// this is a weird thing that does a quasi-dynamic programmingish
		// thing in order to figure out vertical lines and then use that
		// to split up lines 
		
		lines = [].concat.apply([], split_lines(lines, swt).map(function(groups){
			return (groups.length == 1) ? groups : groups.map(wrap_lines)
		})).filter(function(e){
			return e.lettercount > 1
		})
		
		console.timeEnd("split lines")


		function mean(arr){
			for(var s = 0, i = 0; i < arr.length; i++) s += arr[i];
			return s / arr.length;
		}

		function stdev(arr){
			for(var s = 0, ss = 0, i = 0; i < arr.length; i++){
				s += arr[i]
				ss += arr[i] * arr[i]
			}
			return Math.sqrt((ss - s * s / arr.length) / (arr.length - 1))
		}

		lines = lines.map(function(line){
			if(line.letters.length < 7) return line;

			var heights = line.letters.slice(1, -1).map(function(e){ return e.height })
			var avg = mean(heights)

			// this might be a bad idea
			var heights = line.letters.slice(1, -1)
				.map(function(e){ return e.height })
				.filter(function(e){ return e > avg })
			var avg = mean(heights)

			var std = Math.max(1, stdev(heights))

			if(avg < 10) return line;

			// if(first.)
			var letters = line.letters;
			if((letters[0].height - avg) / std > 3){
				letters = letters.slice(1)
			}

			if((letters[letters.length - 1].height - avg) / std > 3){
				letters = letters.slice(0, -1)
			}

			// if(Math.abs(line.angle) - Math.abs(measure_angle(line.letters.slice(1))) > 0.01){
			// 	letters = letters.slice(1)
			// }

			if(letters.length < line.letters.length){
				return wrap_lines(letters)
			}


			return line;



			// return wrap_lines(line.letters.filter(function(e){

			// 	cdf(e.height, mean, std * std)

			// 	console.log('z score' + ((e.height - mean) / std))
			// 	console.log('std ' + std)

			// 	// if(std > 3 && ((e.height - mean) / std) > 0.7){
			// 	// 	return false
			// 	// }
			// 	return true
			// }))
			// return line
		}).filter(function(e){ return e })

		// if(params.debug) visualize_matrix(marker, lines);

		lines = lines.filter(function(line){
			// if(line.letters.length < params.letter_thresh) return false;
			// if(line.width <= line.height * params.elongate_ratio) return false;
			if(Math.abs(line.angle / line.lettercount) > 0.07) return false;
			return true
		})

		if(typeof window == 'object' && params.debug){
			blargh = visualize_matrix(marker, lines);
			lines.forEach(function(line){
				blargh.beginPath()
				var colors = ['green', 'blue', 'red', 'purple', 'orange', 'yellow']
				blargh.strokeStyle = colors[Math.floor(colors.length * Math.random())]
				blargh.lineWidth = 3

				line.letters
					.sort(function(a, b){ return a.cx - b.cx })
					.forEach(function(letter){
						blargh.lineTo(letter.cx, letter.cy)
					})

				blargh.stroke()
			})
		}
		
		if(params.debug) visualize_matrix(marker, lines);

		console.timeEnd('total region extraction')
		console.groupEnd()

		// letter shape is more useful than like the alternative
		lines.forEach(function(line){
			line.letters.forEach(function(letter){
				var contour = [];
				for(var i = 0; i < letter.contours.length; i++){
					var p = letter.contours[i]
					var x = p % width, y = Math.floor(p / width);
					contour.push((x - letter.x0) + (y - letter.y0) * (letter.x1 - letter.x0 + 1))
				}
				delete letter.contours;
				letter.shape = contour;
			})
		})
		return lines;

	}

	var lines = extract_regions(-1).concat(extract_regions(1));

	lines.sort(function(a, b){ return a.cy - b.cy }) // lets sort the lines top to bottom

	return lines;
}

// canny & sobel dx/dy
function raw_swt(img_canny, img_dxdy, params){
	var max_stroke = params.max_stroke, // maximum stroke width
		direction = params.direction,
		width = img_canny.cols,
		height = img_canny.rows;

	// nonzero Math.min function, if a is zero, returns b, otherwise minimizes
	function nzmin(a, b){
		if(a === 0) return b;
		if(a < b) return a;
		return b;
	}
	
	var strokes = [];
	var swt = new jsfeat.matrix_t(width, height, jsfeat.U8C1_t)
	
	console.time("first pass")
	// first pass of stroke width transform 
	for(var i = 0; i < width * height; i++){
		if(img_canny.data[i] != 0xff) continue; // only apply on edge pixels

		var itheta = Math.atan2(img_dxdy.data[(i<<1) + 1], img_dxdy.data[i<<1]); // calculate the image gradient at this point by sobel
		var ray = [i];
		var step = 1;
		
		var ix = i % width, iy = Math.floor(i / width);
		while(step < max_stroke){
			// extrapolate the ray outwards depending on search direction
			// libccv is particularly clever in that it uses 
			// bresenham's line drawing algorithm to pick out
			// the points along the line and also checks 
			// neighboring pixels for corners

			var jx = Math.round(ix + Math.cos(itheta) * direction * step);
			var jy = Math.round(iy + Math.sin(itheta) * direction * step);
			step++;
			if(jx < 0 || jy < 0 || jx > width || jy > height) break;
			var j = jy * width + jx;
			ray.push(j)
			if(img_canny.data[j] != 0xff) continue;
			// calculate theta for this ray since we've reached the other side
			var jtheta = Math.atan2(img_dxdy.data[(j<<1) + 1], img_dxdy.data[j<<1]); 
			
			if(Math.abs(Math.abs(itheta - jtheta) - Math.PI) < Math.PI / 2){ // check if theta reflects the starting angle approximately
				strokes.push(i)
				var sw = Math.sqrt((jx - ix) * (jx - ix) + (jy - iy) * (jy - iy)) // derive the stroke width
				for(var k = 0; k < ray.length; k++){ // iterate rays and set points along ray to minimum stroke width
					swt.data[ray[k]] = nzmin(swt.data[ray[k]], sw) // use nzmin because it's initially all 0's
				}
			}
			break;
		}
	}
	console.timeEnd("first pass")
	console.time("refinement pass")

	// second pass, refines swt values as median
	for(var k = 0; k < strokes.length; k++){
		var i = strokes[k];
		var itheta = Math.atan2(img_dxdy.data[(i<<1) + 1], img_dxdy.data[i<<1]);
		var ray = [];
		var widths = []
		var step = 1;

		var ix = i % width, iy = Math.floor(i / width);
		while(step < max_stroke){
			var jx = Math.round(ix + Math.cos(itheta) * direction * step);
			var jy = Math.round(iy + Math.sin(itheta) * direction * step);
			step++;
			var j = jy * width + jx;
			// record position of the ray and the stroke width there
			widths.push(swt.data[j])
			ray.push(j)			
			// stop when the ray is terminated
			if(img_canny.data[j] == 0xff) break;
		}
		var median = widths.sort(function(a, b){return a - b})[Math.floor(widths.length / 2)];
		// set the high values to the median so that corners are nice
		for(var j = 0; j < ray.length; j++){
			swt.data[ray[j]] = nzmin(swt.data[ray[j]], median)
		}
		// swt.data[ray[0]] = 0
		// swt.data[ray[ray.length - 1]] = 0
	}

	console.timeEnd("refinement pass")
	
	return {
		swt: swt,
		strokes: strokes
	}
}



function coarse_lines(letters, params){
	console.time("form pairs")
	
	// note that in this instance, it might not actually be necessary
	// to use a heap queue because it turns out that we basically compute
	// a list of elements and then process them, and we dont stick things
	// back onto the queue after processing them so we could probably just
	// get by with making an array and sorting it

	// also it might not be necessary to use the find union algorithm, instead
	// we could just keep track of each element's group number and also keep track 
	// of all the groups because at each merge decision we need to access a list
	// of the elements in the relevant groups anyway, so in this case the 
	// performance benefit of the asymptotically inverse ackermann are probably
	// all but lost

	// in addition all those chain merging weights, there should also be something
	// which prioritizes merging lines which are of similar angles rather than
	// introducing a turn.

	// var pair_queue = new HeapQueue(function(a, b){ return a.dist - b.dist })
	var pair_queue = []

	// for(var i = 0; i < letters.length; i++) letters[i].index = i;

	for(var i = 0; i < letters.length; i++){
		var li = letters[i];
		var min_dist = Infinity,
			min_j = null;
		for(var j = 0; j < letters.length; j++){
			if(j == i) continue;

			var lj = letters[j];

			var ratio = li.thickness / lj.thickness;
			if(ratio > params.thickness_ratio || ratio < 1 / params.thickness_ratio) continue;

			if(Math.max(li.height, lj.height) / Math.min(li.height, lj.height) > params.height_ratio) continue;

			if(Math.max(li.width, lj.width) / Math.min(li.width, lj.width) > 10) continue;

			if(li.x0 < lj.x0 && li.x1 > lj.x1) continue; // one is entirely consumed by another
			if(lj.x0 < li.x0 && lj.x1 > li.x1) continue; // one is entirely consumed by another

			var right = (li.x1 > lj.x1) ? li : lj,
				left = (li.x1 > lj.x1) ? lj : li;

			var w = Math.max(0, Math.max(right.x0, left.x0) - Math.min(right.x1, left.x1)),
				h = Math.max(0, Math.max(right.y0, left.y0) - Math.min(right.y1, left.y1));

			// if(w > 2 * Math.max(Math.min(left.height, left.width), Math.min(right.height, right.width))) continue;
			// if(h > 10) continue; // 0 would be safer but 10 allows super extreme angles
			var dy = right.cy - left.cy, 
				dx = right.cx - left.cx;

			var slope = dy / dx
			if(Math.abs(slope) > 1) continue; // cap the max slope


			var dist = w * w + h * h;
			if(dist < min_dist){
				min_dist = dist;
				min_j = lj
			}
		}
		// console.log(min_j, li, dist)
		if(min_j){
			var lj = min_j;
			var right = (li.x1 > lj.x1) ? li : lj,
				left = (li.x1 > lj.x1) ? lj : li;
			pair_queue.push({
				left: left,
				right: right,
				dist: min_dist
			})
		}

	}
	console.log("got shit", pair_queue)


	pair_queue.sort(function(a, b){
		return a.dist - b.dist
	})
	console.timeEnd("form pairs")
	console.time("create lines")

	var groups = []
	for(var i = 0; i < letters.length; i++){
		var letter = letters[i]
		letter.group = groups.length
		groups.push({
			members: [letter]
		})
	}

	var derp = visualize_matrix(params.marker, letters)
	for(var i = 0; i < pair_queue.length; i++){
		var pair = pair_queue[i]
		derp.beginPath()
		var colors = ['green', 'blue', 'red', 'purple', 'orange', 'yellow']
		derp.strokeStyle = colors[Math.floor(colors.length * Math.random())]
		derp.lineWidth = 1
		// if(pair.left.x1 > 100) continue; 
		// if(pair.right.x1 > 100) continue; 
		// derp.moveTo(pair.left.x0 + pair.left.width * Math.random(), pair.left.y0 + Math.random() * pair.left.height)

		// derp.lineTo(pair.right.x0 + pair.right.width * Math.random(), pair.right.y0 + Math.random() * pair.right.height)
		derp.moveTo(pair.left.cx, pair.left.cy)
		derp.lineTo(pair.right.cx, pair.right.cy)
		derp.stroke()

	}

	var total_length = pair_queue.length;

	while(pair_queue.length){
		var pair = pair_queue.shift()
		
		var left_group = pair.left.group,
			right_group = pair.right.group;
			
		if(left_group == right_group) continue;

		var lca = groups[left_group].members,
			rca = groups[right_group].members;
		


		// if(!(lca[0] == pair.left || lca[lca.length - 1] == pair.left)) continue;
		// if(!(rca[0] == pair.right || rca[rca.length - 1] == pair.right)) continue;

		// if(!( && (rca[0] == pair.right || rca[rca.length - 1] == pair.right))) continue; 

		var merged = lca.concat(rca).sort(function(a, b){ return a.x1 - b.x1 })

		for(var i = 0; i < lca.length; i++)
			lca[i].group = right_group;

		groups[right_group].members = merged		
		// groups[right_group].slope = dy / dx;

		// var merp = visualize_matrix(params.marker)
		// groups.filter(function(e){
		// 	return e
		// }).map(function(e){
		// 	merp.beginPath()
		// 	var colors = ['green', 'blue', 'red', 'purple', 'orange', 'yellow']
		// 	merp.strokeStyle = colors[Math.floor(colors.length * Math.random())]
		// 	merp.lineWidth = 3
		// 	e.members.forEach(function(letter){
		// 		merp.lineTo(letter.cx, letter.cy)
		// 		if(e.members.length > 1){
		// 			merp.strokeRect(letter.x0, letter.y0, letter.width, letter.height)	
		// 		}
				
		// 	})
		// 	merp.stroke()
		// })

		groups[left_group] = null
	}

	console.timeEnd("create lines")


	return groups.filter(function(e){
		return e
	}).map(function(e){
		return e.members
	})


}



function find_lines(letters, params){
	console.time("form pairs")
	
	// note that in this instance, it might not actually be necessary
	// to use a heap queue because it turns out that we basically compute
	// a list of elements and then process them, and we dont stick things
	// back onto the queue after processing them so we could probably just
	// get by with making an array and sorting it

	// also it might not be necessary to use the find union algorithm, instead
	// we could just keep track of each element's group number and also keep track 
	// of all the groups because at each merge decision we need to access a list
	// of the elements in the relevant groups anyway, so in this case the 
	// performance benefit of the asymptotically inverse ackermann are probably
	// all but lost

	// in addition all those chain merging weights, there should also be something
	// which prioritizes merging lines which are of similar angles rather than
	// introducing a turn.

	// var pair_queue = new HeapQueue(function(a, b){ return a.dist - b.dist })
	var pair_queue = []

	// for(var i = 0; i < letters.length; i++) letters[i].index = i;

	for(var i = 0; i < letters.length; i++){
		var li = letters[i];
		for(var j = i + 1; j < letters.length; j++){
			var lj = letters[j];

			var ratio = li.thickness / lj.thickness;
			if(ratio > params.thickness_ratio || ratio < 1 / params.thickness_ratio) continue;

			if(Math.max(li.height, lj.height) / Math.min(li.height, lj.height) > params.height_ratio) continue;

			if(Math.max(li.width, lj.width) / Math.min(li.width, lj.width) > 10) continue;

			if(li.x0 < lj.x0 && li.x1 > lj.x1) continue; // one is entirely consumed by another
			if(lj.x0 < li.x0 && lj.x1 > li.x1) continue; // one is entirely consumed by another

			var right = (li.x1 > lj.x1) ? li : lj,
				left = (li.x1 > lj.x1) ? lj : li;

			// var woverlap = Math.max(0, Math.min(right.x1, left.x1) - Math.max(right.x0, left.x0)),
			// 	hoverlap = Math.max(0, Math.min(right.y1, left.y1) - Math.max(right.y0, left.y0));

			var w = Math.max(0, Math.max(right.x0, left.x0) - Math.min(right.x1, left.x1)),
				h = Math.max(0, Math.max(right.y0, left.y0) - Math.min(right.y1, left.y1));


			// var h_dist = Math.max(0, right.x0 - left.x1)
			if(w > 2 * Math.max(Math.min(left.height, left.width), Math.min(right.height, right.width))) continue;

			// if(w > 2 * Math.max(Math.sqrt(left.width * left.height), Math.sqrt(right.width * right.height))) continue;
			// if(w > 2 * Math.min(Math.max(left.height, left.width), Math.max(right.height, right.width))) continue;

			if(h > 10) continue; // 0 would be safer but 10 allows super extreme angles

			// if(Math.max(li.markweight, lj.markweight)/Math.min(li.markweight, lj.markweight) > 4) continue;

			var dy = right.cy - left.cy, 
				dx = right.cx - left.cx;

			var slope = dy / dx

			if(Math.abs(slope) > 1) continue; // cap the max slope


			// if(Math.max(right.contours.length, left.contours.length) / Math.min(right.contours.length, left.contours.length) > 3) continue; 

			// var h_align = Math.min(Math.abs(left.y0 - right.y0), Math.abs(left.y1 - right.y1), Math.abs(left.cy - right.cy))

			pair_queue.push({
				left: left,
				right: right,
				// dist: w
				// this is meant to bias things toward flatness
				// but this isn't necessarily good because it also flattens
				// things that aren't flat
				// dist: w * w + h
				// dist: Math.sqrt(10 * dy * dy + Math.pow(dx + w, 2)) // euclidean distance ftw
				// dist: Math.sqrt(10 * dy * dy + dx * dx) // euclidean distance ftw
				dist: Math.sqrt(20 * Math.pow(dy + h, 2) + Math.pow(dx + w, 2)) // new thingy

				// dist: Math.sqrt(10 * dy * dy + w * w) // frankendistance
				// this minimizes the x distance and also puts a weight on the y distance
			})
		}
	}


	pair_queue.sort(function(a, b){
		return a.dist - b.dist
	})
	console.timeEnd("form pairs")
	console.time("create lines")

	var groups = []
	for(var i = 0; i < letters.length; i++){
		var letter = letters[i]
		letter.group = groups.length
		groups.push({
			members: [letter]
		})
	}

	// var derp = visualize_matrix(params.marker, letters)
	// for(var i = 0; i < pair_queue.length; i++){
	// 	var pair = pair_queue[i]
	// 	derp.beginPath()
	// 	var colors = ['green', 'blue', 'red', 'purple', 'orange', 'yellow']
	// 	derp.strokeStyle = colors[Math.floor(colors.length * Math.random())]
	// 	derp.lineWidth = 1
	// 	if(pair.left.x1 > 100) continue; 
	// 	if(pair.right.x1 > 100) continue; 
	// 	derp.moveTo(pair.left.x0 + pair.left.width * Math.random(), pair.left.y0 + Math.random() * pair.left.height)
	// 	derp.lineTo(pair.right.x0 + pair.right.width * Math.random(), pair.right.y0 + Math.random() * pair.right.height)
	// 	derp.stroke()

	// }

	// var forest = new UnionFind(letters.length)

	// this is like the sum of the absolute value of the second derivative
	// of the center of each letter sorted by x, if two runs of letters
	// get merged together and overlap, then it'll register as a pretty 
	// big ziggometric spike which means that we can exclude those
	// and the second derivative means 
	function zigometer(set){	
		var v_overlap = 2; // this is the allowable vertical extent

		if(set.length < 3) return 0; // cant calculate discrete 2nd deriv of 2 points
		set.sort(function(a, b){ return a.x1 - b.x1 }) // im debating whether this is a better metric than cx
		var last = set[0], lastdy, sigddy = 0;
		for(var i = 1; i < set.length; i++){
			// var dy = set[i].cy - last.cy;
			// var dy = Math.max(0, Math.abs(set[i].cy - last.cy) - Math.max(set[i].height, last.height));
			var dy =  Math.max(v_overlap, Math.max(last.y0, set[i].y0) - Math.min(last.y1, set[i].y1)) - v_overlap
			if(i > 1) sigddy += Math.abs(dy - lastdy);
			lastdy = dy
			last = set[i]
		}
		return 1000 * sigddy
	}

	function zigometer_strict(set){	
		var v_overlap = 2; // this is the allowable vertical extent

		if(set.length < 3) return 0; // cant calculate discrete 2nd deriv of 2 points
		set.sort(function(a, b){ return a.x1 - b.x1 }) // im debating whether this is a better metric than cx
		var last = set[0], lastdy, sigddy = 0;
		for(var i = 1; i < set.length; i++){
			var dy = set[i].cy - last.cy;
			// var dy = Math.max(0, Math.abs(set[i].cy - last.cy) - Math.max(set[i].height, last.height));
			// var dy =  Math.max(v_overlap, Math.max(last.y0, set[i].y0) - Math.min(last.y1, set[i].y1)) - v_overlap
			if(i > 1) sigddy += Math.abs(dy - lastdy);
			lastdy = dy
			last = set[i]
		}
		return 1000 * sigddy
	}
	function measure_angle(letters){
		if(letters.length == 1) return 0;

		var slopes = []
		for(var i = 0; i < letters.length; i++){
			var li = letters[i];
			for(var j = 0; j < i; j++){
				var lj = letters[j];
				slopes.push((li.cy - lj.cy) / (li.cx - lj.cx))
			}
		}
		return Math.atan(slopes.sort(function(a, b){ return a - b })[Math.floor(slopes.length/2)])
	}
	// function measure_angle(letters){
	// 	if(letters.length == 1) return 0;

	// 	var sumx = 0, sumx2 = 0, sumxy = 0, sumy = 0, sumy2 = 0;
	// 	for(var i = 0; i < letters.length; i++){
	// 		var letter = letters[i];
	// 		sumx += letter.cx; sumy += letter.cy;
	// 		sumx2 += letter.cx * letter.cx; sumy2 += letter.cy * letter.cy;
	// 		sumxy += letter.cx * letter.cy;
	// 	}
	// 	var n = letters.length;
	// 	var dy = (n * sumxy - sumx * sumy);
	// 	var dx = (n * sumx2 - sumx * sumx);
	// 	var yi = (sumy * sumx2 - sumx * sumxy) / (n * sumx2 - sumx * sumx);
	// 	var r = (sumxy - sumx * sumy / n) / Math.sqrt((sumx2 - sumx*sumx/n) * (sumy2 - sumy*sumy/n));
	// 	return Math.atan2(dy, dx)
	// }
	function bounding_box(set){
		var x0 = set[0].x0, y0 = set[0].y0,
			x1 = set[0].x1, y1 = set[0].y1;
		for(var i = 1; i < set.length; i++){
			x0 = Math.min(x0, set[i].x0)
			y0 = Math.min(y0, set[i].y0)
			x1 = Math.max(x1, set[i].x1)
			y1 = Math.max(y1, set[i].y1)
		}
		return {x0: x0, y0: y0, x1: x1, y1: y1, width: x1 - x0, height: y1 - y0}
	}
	function intersects(a, b){
		var width = Math.min(a.x1, b.x1) - Math.max(a.x0, b.x0),
			height = Math.min(a.y1, b.y1) - Math.max(a.y0, b.y0);
		var min_area = Math.min((a.x1 - a.x0) * (a.y1 - a.y0), (b.x1 - b.x0) * (b.y1 - b.y0))
		return (width > 0 && height > 0) && (width * height) > 0.3 * min_area
	}
	var total_length = pair_queue.length;

	while(pair_queue.length){
		var pair = pair_queue.shift()
		
		var left_group = pair.left.group,
			right_group = pair.right.group;
			
		if(left_group == right_group) continue;

		var lca = groups[left_group].members,
			rca = groups[right_group].members;
		


		// if(!(lca[0] == pair.left || lca[lca.length - 1] == pair.left)) continue;
		// if(!(rca[0] == pair.right || rca[rca.length - 1] == pair.right)) continue;

		// if(!( && (rca[0] == pair.right || rca[rca.length - 1] == pair.right))) continue; 
		var langle = measure_angle(lca),
			rangle = measure_angle(rca);

		var merged = lca.concat(rca).sort(function(a, b){ return a.x1 - b.x1 })

		if(lca.length > 1 || rca.length > 1){
			var zigtotes = zigometer(merged) / (lca.length + rca.length);
			var angtotes = measure_angle(merged)
			
			if(Math.abs(angtotes) > 0.1 + Math.abs(langle) + Math.abs(rangle)) continue;

			if(zigtotes > 0) continue;	

			var r_bb = bounding_box(rca),
				l_bb = bounding_box(lca);

			if(intersects(r_bb, l_bb)) continue;

			// var dy = Math.abs((r_bb.y0 / 2 + r_bb.y1 / 2) - (l_bb.y0 / 2 + l_bb.y1 / 2))

			// var y_overlap = Math.min(r_bb.y1, l_bb.y1) - Math.max(r_bb.y0, l_bb.y0);
			// if(y_overlap < 0) continue;

			// var frac_overlap = y_overlap / Math.min(l_bb.y1 - l_bb.y0, r_bb.y1 - r_bb.y0)
			
			// if(frac_overlap < 1 - 0.1 / Math.max(lca.length, rca.length)) continue;

			// var l_dim = Math.max.apply(Math, lca.map(function(e){ return e.contours.length }))
			// var r_dim = Math.max.apply(Math, rca.map(function(e){ return e.contours.length }))

			// var ratio = Math.max(r_dim, l_dim) / Math.min(r_dim, l_dim)
			// if(ratio > 1 + 10 / Math.max(lca.length, rca.length)) continue;


			var l_height = Math.max.apply(Math, lca.map(function(e){ return e.height }))
			var r_height = Math.max.apply(Math, rca.map(function(e){ return e.height }))
			// if(dy / Math.max(l_height, r_height) > 0.1){
			var ratio = Math.max(r_height, l_height) / Math.min(r_height, l_height)

			if(ratio > 1.5 + 10 / Math.max(lca.length, rca.length)) continue;

			// var l_width = Math.max.apply(Math, lca.map(function(e){ return e.width }))
			// var r_width = Math.max.apply(Math, rca.map(function(e){ return e.width }))

			// var ratio = Math.max(r_width, l_width) / Math.min(r_width, l_width)

			// if(ratio > 1.5 + 20 / Math.max(lca.length, rca.length)) continue;

	
			// }
			

		}
		// var sumx = 0, sumx2 = 0, sumxy = 0, sumy = 0, sumy2 = 0;

		// for(var i = 0; i < merged.length; i++){
		// 	var letter = merged[i];
		// 	sumx += letter.cx; sumy += letter.cy;
		// 	sumx2 += letter.cx * letter.cx; sumy2 += letter.cy * letter.cy;
		// 	sumxy += letter.cx * letter.cy;
		// }

		// var n = merged.length;
		// var dy = (n * sumxy - sumx * sumy);
		// var dx = (n * sumx2 - sumx * sumx);
		// var yi = (sumy * sumx2 - sumx * sumxy) / (n * sumx2 - sumx * sumx);
		// var r = (sumxy - sumx * sumy / n) / Math.sqrt((sumx2 - sumx*sumx/n) * (sumy2 - sumy*sumy/n));

		// console.log(dy/dx, groups[left_group].slope, groups[right_group].slope)
		// if(Math.abs(dy/dx) - 0.01 > 2 * Math.max(Math.abs(groups[right_group].slope), Math.abs(groups[left_group].slope))) continue;

		for(var i = 0; i < lca.length; i++)
			lca[i].group = right_group;

		groups[right_group].members = merged		
		// groups[right_group].slope = dy / dx;

		// var merp = visualize_matrix(params.marker)
		// groups.filter(function(e){
		// 	return e
		// }).map(function(e){
		// 	merp.beginPath()
		// 	var colors = ['green', 'blue', 'red', 'purple', 'orange', 'yellow']
		// 	merp.strokeStyle = colors[Math.floor(colors.length * Math.random())]
		// 	merp.lineWidth = 3
		// 	e.members.forEach(function(letter){
		// 		merp.lineTo(letter.cx, letter.cy)
		// 		if(e.members.length > 1){
		// 			merp.strokeRect(letter.x0, letter.y0, letter.width, letter.height)	
		// 		}
				
		// 	})
		// 	merp.stroke()
		// })

		groups[left_group] = null
	}

	console.timeEnd("create lines")


	return groups.filter(function(e){
		return e
	}).map(function(e){
		return e.members
	})


}



function filter_regions(regions, src){
	// what is the radius of the structuring element for our morphological dilation
	var dilrad = 4;
	// how many pixels to include outside the region in our color filtering
	var xpad = 30,
		ypad = 20;

	var width = src.width,
		height = src.height;

	var marker = new jsfeat.matrix_t(width, height, jsfeat.U8C1_t);
	var dilation = new jsfeat.matrix_t(width, height, jsfeat.U8C1_t);
	

	console.time('morphological dilation')

	for(var a = regions.length - 1; a >= 0; a--){
		for(var b = regions[a].letters.length - 1; b >= 0; b--){
			var c = regions[a].letters[b].contours;
			for(var i = c.length - 1; i >= 0; i--){
				for(var dx = -dilrad; dx <= dilrad; dx++){
					for(var dy = -dilrad; dy <= dilrad; dy++){
						dilation.data[c[i] + dx + dy * width] = 1
					}
				}
			}
		}
	}
	// set thing in there to 2 so that we know which ones are inside
	for(var a = regions.length - 1; a >= 0; a--){
		for(var b = regions[a].letters.length - 1; b >= 0; b--){
			var c = regions[a].letters[b].contours;
			for(var i = c.length - 1; i >= 0; i--){
				dilation.data[c[i]] = 2
			}
		}
	}

	console.timeEnd('morphological dilation')

	
	
	var pixmap = new Uint16Array(width * height)
	var intoct = new Uint32Array(16 * 16 * 16)
	var extoct = new Uint32Array(16 * 16 * 16)
	var zeroes = new Uint32Array(16 * 16 * 16)

	var labtab = {};

	console.time("color filter")

	for(var region_num = 0; region_num < regions.length; region_num++){

		
		var line = regions[region_num]
		// console.log(line)
		intoct.set(zeroes)
		extoct.set(zeroes)
		
		var x0 = Math.max(0, line.x0 - xpad),
			x1 = Math.min(width, line.x1 + xpad),
			y0 = Math.max(0, line.y0 - ypad),
			y1 = Math.min(height, line.y1 + ypad);

		for(var y = y0; y < y1; y++){
			for(var x = x0; x < x1; x++){
				var p = x + y * width;

				var rgb_color = Math.floor(src.data[4 * p] / 8) + 
								Math.floor(src.data[4 * p + 1] / 8) * 32 + 
								Math.floor(src.data[4 * p + 2] / 8) * 1024;
				// here we round the color kinda to speed up rgb->lab conversion
				// because the conversion is kinda slow

				if(!(rgb_color in labtab)){

					var lab = rgb2lab([src.data[4 * p], src.data[4 * p + 1], src.data[4 * p + 2]])
					// var color =     Math.floor((2 * lab[0] + 30) / 16) + 
					// 				Math.floor((lab[1] + 130) / 16) * 16 + 
					// 				Math.floor((lab[2] + 130) / 16) * 256
					var color = Math.round((2 * lab[0] + 40) / 16) |
								Math.round((lab[1] + 128) / 16) << 4 |
								Math.round((lab[2] + 128) / 16) << 8;

					labtab[rgb_color] = color					
				}else{
					var color = labtab[rgb_color]
				}

				pixmap[p] = color
				if(dilation.data[p] === 1){
					extoct[color]++
					extoct[color+16]++
					extoct[color-16]++
					extoct[color+256]++
					extoct[color-256]++
					extoct[color+1]++
					extoct[color-1]++
				}else if(dilation.data[p] === 2){
					intoct[color]++
					intoct[color+16]++
					intoct[color-16]++
					intoct[color+256]++
					intoct[color-256]++
					intoct[color+1]++
					intoct[color-1]++
				}
			}
		}

		// var invmap = {}
		// for(var i in labtab){
		// 	invmap[labtab[i]] = [(i % 32) * 8, (Math.floor(i / 32) % 32) * 8, (Math.floor(i / 1024)) * 8 ]
		// }

		// var merp = document.createElement('canvas')
		// merp.width = 256
		// merp.height = 256
		// merp.style.background = 'url(data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAMAAAADCAYAAABWKLW/AAAAHUlEQVQIW2NkgILFixf/ZwSxQYzY2FhGRhgDJAgAwf8K7NQ1zbQAAAAASUVORK5CYII=)'
		// document.body.appendChild(merp)
		// merp = merp.getContext('2d')
		
		
		
		for(var y = y0; y < y1; y++){
			for(var x = x0; x < x1; x++){
				var p = x + y * width;	
				var color = pixmap[p]
				if(intoct[color] / (1 + extoct[color]) > 3){
					marker.data[p] = Math.min(255, 2 * (intoct[color] / (1 + extoct[color])))
					
				}

				// var l = (color % 16) * 16, 
				// 	a = (Math.floor(color / 16) % 16) * 16, 
				// 	b = (Math.floor(color / 256)) * 16;
				// if(intoct[color] / (1 + extoct[color]) > 2){
				// 	merp.fillStyle = 'rgb(' + invmap[color].join(',') + ')'
				// 	merp.fillRect(l, b, 16, 16)
				// }else{

				// 	merp.strokeStyle = 'rgb(' + invmap[color].join(',') + ')'
				// 	merp.lineWidth = 3
				// 	merp.strokeRect(l + merp.lineWidth, b + merp.lineWidth, 16 - 2 * merp.lineWidth, 16 - 2 * merp.lineWidth)
				// }
			}
		}
		
	}

	console.timeEnd('color filter')

	return marker
}



function split_lines(regions, swt){
	var width = swt.cols,
		height = swt.rows;
	
	// this is a kind of constrained linear crawl upwards and downwards to 
	// detect vertical lines that are used to separate content

	return regions.map(function(line){
		var buf = [line.letters[0]], groups = [];
		for(var i = 0; i < line.letters.length - 1; i++){
			var cur = line.letters[i],
				next = line.letters[i + 1];
			
			if(next.x0 - cur.x1 > Math.sqrt(Math.min(next.area, cur.area))){
				var streak = -1, separators = 0, y = Math.floor(cur.cy / 2 + next.cy / 2);
				var goal = 3 * Math.max(cur.height, next.height)
				for(var x = cur.x1; x < next.x0; x++){
					var n = y * width + x;
					if(swt.data[n] > 0){
						if(streak < 0) streak = x;
					}else{
						if(streak > 0){
							var mid = Math.floor(x / 2 + streak / 2), explored = 0;
							for(var t = 0; t < goal; t++){
								var k = (y + t) * width + mid;
								if(swt.data[k] > 0){
									explored++;
								}else if(swt.data[k + 1] > 0){
									mid++; explored++
								}else if(swt.data[k - 1] > 0){
									mid--; explored++
								}else break;
							}
							var mid = Math.floor(x / 2 + streak / 2);
							for(var t = 0; t < goal; t++){
								var k = (y - t) * width + mid;
								if(swt.data[k] > 0){
									explored++;
								}else if(swt.data[k + 1] > 0){
									mid++; explored++
								}else if(swt.data[k - 1] > 0){
									mid--; explored++
								}else break;
							}
							if(explored > goal) separators++;
						}
						streak = -1
					}
				}
				if(separators > 0){ // break it off
					groups.push(buf)
					buf = []
				}
			}
			buf.push(next)
		}
		groups.push(buf)
		// return groups
		return (groups.length == 1) ? [line] : groups
	})
}


function exclude_occlusions(letters, width, height, params){
	console.time("excluding occlusions")
	
	var buffer = new jsfeat.matrix_t(width, height, jsfeat.S32_t | jsfeat.C1_t); 

	for(var i = 0; i < letters.length; i++){
		var contour = letters[i].contours;
		for(var j = 0; j < contour.length; j++){
			buffer.data[contour[j]] = i + 1;
		}
	}


	// var removed = [];

	// letters.map(function(letter, i){
	// 	var another = [];
	// 	var occlusions = 0;
	// 	for(var x = letter.x0; x < letter.x1; x++){
	// 		for(var y = letter.y0; y < letter.y1; y++){
	// 			var group = buffer.data[x + width * y];
	// 			if(group && group != i + 1){
	// 				occlusions++;
	// 				if(another.indexOf(group) == -1){
	// 					another.push(group)
	// 				}
	// 			}
	// 		}
	// 	}
	// 	// if it has few occlusions, it gets a free pass
	// 	if(another.length < params.letter_occlude_thresh) return false;

	// 	return [i, another]
	// })
	// .filter(function(e){ return e })
	// .sort(function(a, b){
	// 	// kill these things in order of area ascending
	// 	// return a.area - b.area
	// 	// return a[1].length - b[1].length
	// 	return a[0].area - b[0].area
	// }).forEach(function(e){
	// 	var i = e[0],
	// 		another = e[1];
		
	// 	var remaining = another.filter(function(e){ return removed.indexOf(e) == -1 }).length;
	// 	if(remaining >= params.letter_occlude_thresh){
	// 		removed.push(i)
	// 	}
	// });

	// return letters.filter(function(letter, i){
	// 	return removed.indexOf(i) == -1
	// })


	return letters.filter(function(letter, i){
		var another = [];
		var occlusions = 0;
		for(var x = letter.x0; x < letter.x1; x++){
			for(var y = letter.y0; y < letter.y1; y++){
				var group = buffer.data[x + width * y];
				if(group && group != i + 1){
					occlusions++;
					if(another.indexOf(group) == -1){
						another.push(group)
					}
				}
			}
		}
		if(another.length >= params.letter_occlude_thresh) return false;
		// if(occlusions > 120) return false;
		// if(occlusions / letter.contours.length > params.occlusion_ratio) return false;
		return true;
	})
	console.timeEnd("excluding occlusions")
}

function count_bits(v){
	var c = 0; // count the number of bits set in v
	for (c = 0; v; c++) {
		v &= v - 1; // clear the least significant bit set
	}
	return c
}

// 

function visualize_matrix(mat, letters){
	var c = new Canvas(mat.cols, mat.rows);
	var cx = c.getContext('2d')
	var out = cx.createImageData(mat.cols, mat.rows);

	for(var i = 0; i < mat.rows * mat.cols; i++){
		out.data[i * 4 + 3] = 255
		if(mat.data[i] == 1){
			out.data[i * 4] = 255
		}else{
			out.data[i * 4] = out.data[i * 4 + 1] = out.data[i * 4 + 2] = 30 * mat.data[i]	
		}
	}
	cx.putImageData(out, 0, 0)
	
	if(letters){
		cx.strokeStyle = 'red'
		for(var i = 0; i < letters.length; i++){
			var letter = letters[i];
			cx.strokeRect(letter.x0 + .5, letter.y0 + .5, letter.width, letter.height)
		}

		if(letters[0] && letters[0].letters){
			// hey look not actually letters
			letters.forEach(function(line){
				cx.beginPath()
				var colors = ['green', 'blue', 'red', 'purple', 'orange', 'yellow']
				cx.strokeStyle = colors[Math.floor(colors.length * Math.random())]
				cx.lineWidth = 3

				line.letters
					.sort(function(a, b){ return a.cx - b.cx })
					.forEach(function(letter){
						cx.lineTo(letter.cx, letter.cy)
					})

				cx.stroke()
			})
		}
	}

	broadcast(c.pngStream(), {})
}



function grayscale_matrix(mat){
	var c = new Canvas(mat.cols, mat.rows);
	var cx = c.getContext('2d')
	var out = cx.createImageData(mat.cols, mat.rows);

	for(var i = 0; i < mat.rows * mat.cols; i++){
		out.data[i * 4 + 3] = 255
		out.data[i * 4] = out.data[i * 4 + 1] = out.data[i * 4 + 2] = mat.data[i];
	}
	cx.putImageData(out, 0, 0)
	broadcast(c.pngStream(), {})
}



function do_processing(){
	var img_dir = 'Challenge1_Training_Task12_Images';
	var gt_dir = 'Challenge1_Training_Task1_GT'; // wow such ground truth

	
	fs.readdir(img_dir, function(err, names){
		names.forEach(function(name){
			var groundTruth = fs.readFileSync(path.join(gt_dir, 'gt_' + name.replace(/\.\w+$/, ".txt")), 'utf8')
				.split('\n')
				.map(function(text){ return JSON.parse('[' + text + ']'); })
				.filter(function(e){ return e.length > 4 })
				.map(function(region){
					var a = {
						x0: region[0],
						y0: region[1],
						x1: region[2],
						y1: region[3]
					}
					a.width = a.x1 - a.x0 + 1;
					a.height = a.y1 - a.y0 + 1;
					a.area = a.width * a.height;
					return a
				});

			fs.readFile(path.join(img_dir, name), function(err, file){
				if (err) throw err;

				var img = new Image();
				img.src = file;

  				var canvas = new Canvas(img.width, img.height),
  					ctx = canvas.getContext('2d');

				ctx.drawImage(img, 0, 0);

				console.log(name)
				var width = img.width, height = img.height;

				var img_data = ctx.getImageData(0, 0, width, height);
				// var img_u8 = new jsfeat.matrix_t(width, height, jsfeat.U8C1_t)
				// console.log(img_u8, jsfeat.U8C1_t)
				// jsfeat.imgproc.grayscale(img_data.data, img_u8.data)
				// grayscale_matrix(img_u8)
				var result = partial_swt(img_data, params)

				ctx.strokeStyle = 'red'
				result.forEach(function(line){
					ctx.strokeRect(line.x0, line.y0, line.width, line.height)
				})

				ctx.strokeStyle = 'green'

				var true_positive = 0;


				groundTruth.forEach(function(a){
					
					console.log(a)

					ctx.strokeRect(a.x0, a.y0, a.width, a.height)

					var matches = result.some(function(b){
						var width = Math.min(a.x1, b.x1) - Math.max(a.x0, b.x0),
							height = Math.min(a.y1, b.y1) - Math.max(a.y0, b.y0);
						var val = width > 0 && height > 0 && width * height > 0.8 * a.area;
						if(val) b._true = true;
						return val
					})

					if(matches){
						true_positive++;
						ctx.fillRect(a.x0, a.y0, a.width, a.height)
					}
				})




				// ctx.fillText("TP: " + (true_positive / groundTruth.length),10, 10)

				// var img_dxdy = new jsfeat.matrix_t(width, height, jsfeat.S32C2_t);
				// console.log('flerp', img_u8.data)
				// visualize_matrix(img_u8)
				// jsfeat.imgproc.sobel_derivatives(img_u8, img_dxdy)
				// jsfeat.imgproc.gaussian_blur(img_u8, img_u8, params.kernel_size, 0)
				// jsfeat.imgproc.canny(img_u8, img_u8, params.low_thresh, params.high_thresh)
				console.log('merp')
				// grayscale_matrix(img_u8)

				var recall = true_positive / groundTruth.length,
					precision = result.filter(function(e){ return e._true }).length / Math.max(1, result.length);

				broadcast(canvas.pngStream(), {
					recall: recall,
					precision: precision
				})
			})	
		})
		
	})	
}


// do_processing();
