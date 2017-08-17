<?
	$data = [];
	$class = $_GET['val'];
	if (!$class) { $class = 'bird'; }
	$dataset = "Dataset";
	
	$base_path = "/Users/damianocancemi/PycharmProjects/SMM/Progetto";
	$dirname = $base_path."/".$dataset."/".$class;
	$dir = new DirectoryIterator($dirname);
	foreach ($dir as $item_info) {
		if (!$item_info->isDot() && $item_info->getExtension() == 'jpg') {
			array_push($data, $item_info->getFilename());
		}
	}
?>

<html>
	<head>
		<title>Cancemi Damiano - W82000075</title>
		<link rel="stylesheet" href="//maxcdn.bootstrapcdn.com/bootstrap/3.3.7/css/bootstrap.min.css" integrity="sha384-BVYiiSIFeK1dGmJRAkycuHAHRg32OmUcww7on3RYdg4Va+PmSTsz/K68vbdEjh4u" crossorigin="anonymous">
		<link rel="stylesheet" href="//cdnjs.cloudflare.com/ajax/libs/bootstrap-select/1.12.2/css/bootstrap-select.min.css">
		<link href="css/style.css" rel="stylesheet">
	</head>
	<body>
		<div class="container">
			<div class="logo">
				<img src="images/flickr.png" /><span>classification</span>
			</div>
			<div id="class_button" class="row center-block">
				<select class="selectpicker">
					<option <?=$class == 'bird' ? 'selected' : ''; ?>>Bird</option>
					<option <?=$class == 'mammal' ? 'selected' : ''; ?>>Mammal</option>
				</select>
			</div>
			<div id="gallery" class="row">
				<?
					foreach ($data as $d) {
						echo '
							<div class="col-lg-3 col-sm-4 col-xs-6 item">
								<a title="'.$d.'" href="#">
									<img class="thumbnail img-responsive" src="../'.$dataset."/".$class."/".$d.'">
								</a>
							</div>
						';
					}
				?>
			</div>
		</div>
		<div tabindex="-1" class="modal fade" id="myModal" role="dialog">
			<div class="modal-dialog">
				<div class="modal-content">
					<div class="modal-header">
						<button class="close" type="button" data-dismiss="modal">x</button>
						<h3 class="modal-title">Heading</h3>
					</div>
					<div class="modal-body"></div>
				</div>
			</div>
		</div>
		<script src="//code.jquery.com/jquery-latest.min.js" type="text/javascript"></script>
		<script src="//maxcdn.bootstrapcdn.com/bootstrap/3.3.7/js/bootstrap.min.js" integrity="sha384-Tc5IQib027qvyjSMfHjOMaLkfuWVxZxUPnCJA7l2mCWNIpG9mGCD8wGNIcPD7Txa" crossorigin="anonymous"></script>		
		<script src="//cdnjs.cloudflare.com/ajax/libs/bootstrap-select/1.12.2/js/bootstrap-select.min.js"></script>
		<script>
			$(document).ready(function() {
				$('.thumbnail').click(function(){
					$('.modal-body').empty();
					var title = $(this).parent('a').attr("title");
					$('.modal-title').html(title);
					$($(this).parents('div').html()).appendTo('.modal-body');
					$('#myModal').modal({show:true});
				});
				$('.selectpicker').selectpicker({
					style: 'btn-primary'
				});
			});
			
			$('.selectpicker').on('change', function() {
				var val = $('.selectpicker').val().toLowerCase();
				location.href = '?val='+val;
			});
		</script>
	</body>
</html>