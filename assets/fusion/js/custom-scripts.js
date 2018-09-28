(function($) {
    
  'use strict';


  /**
   * =====================================
   * Function for windows height and width      
   * =====================================
   */
  function windowSize( el ) {
    var result = 0;
    if("height" == el)
        result = window.innerHeight ? window.innerHeight : $(window).height();
    if("width" == el)
      result = window.innerWidth ? window.innerWidth : $(window).width();

    return result; 
  }


  /**
   * =====================================
   * Function for email address validation         
   * =====================================
   */
  function isValidEmail(emailAddress) {
    var pattern = new RegExp(/^(("[\w-\s]+")|([\w-]+(?:\.[\w-]+)*)|("[\w-\s]+")([\w-]+(?:\.[\w-]+)*))(@((?:[\w-]+\.)*\w[\w-]{0,66})\.([a-z]{2,6}(?:\.[a-z]{2})?)$)|(@\[?((25[0-5]\.|2[0-4][0-9]\.|1[0-9]{2}\.|[0-9]{1,2}\.))((25[0-5]|2[0-4][0-9]|1[0-9]{2}|[0-9]{1,2})\.){2}(25[0-5]|2[0-4][0-9]|1[0-9]{2}|[0-9]{1,2})\]?$)/i);
    return pattern.test(emailAddress);
  };


  /**
   * =====================================
   * Function for windows height and width      
   * =====================================
   */
  function deviceControll() {
    if( windowSize( 'width' ) < 768 ) {
      $('body').removeClass('desktop').removeClass('tablet').addClass('mobile');
    }
    else if( windowSize( 'width' ) < 992 ){
      $('body').removeClass('mobile').removeClass('desktop').addClass('tablet');
    }
    else {
      $('body').removeClass('mobile').removeClass('tablet').addClass('desktop');
    }
  }



  function fullpageInitialization( scrollOverflow ) {
    $('#fullpage').fullpage({
      hybrid:             false,
      scrollOverflow:     scrollOverflow,
      scrollBar:          false,
      keyboardScrolling:  true,
      scrollingSpeed:     900,
      verticalCentered:   true,
      easingcss3:         'cubic-bezier(0.785, 0.135, 0.150, 0.860)',
      css3:               true,
      // navigation: true,
      // navigationPosition: 'left',
      // navigationTooltips: ['First page', 'Second page', 'Third and last page'],


      afterLoad: function(anchorLink, index){

        // console.log(index);

        if( index !== 1 ) {
          $('.navbar-top-js').addClass('navbar-home');
          $('#fp-nav').fadeIn(10).addClass('show-nav');
        } 
        else {
          $('.navbar-fixed-top').removeClass('navbar-home');
          $('#fp-nav').removeClass('show-nav');

          setTimeout(function() {
            $('#fp-nav').fadeOut(10);
          }, 600)
        }


        $('.section').each(function() {

          if( $(this).find('.inner-container').height() < windowSize('height') ) {
            $(this).find('.fp-tableCell, .slimScrollDiv, .fp-scrollable').css({
              'height': 'initial'
            })
          }else {
            $(this).find('.fp-tableCell, .slimScrollDiv, .fp-scrollable').css({
              'height': windowSize('height')
            })
          }
        })



        var navigationClickFullPage = '#navbar-nav, #fp-nav';
        $( navigationClickFullPage ).find('a').each( function() {

          if( '#' + index === $(this).attr('href') ) {
            $(this).closest( navigationClickFullPage ).find('.active').removeClass('active');
            $(this).addClass('active');
          }
          
        });


      },
      onLeave: function(index, nextIndex, direction){
        
        var ind             = index - 1,
            nextInd         = nextIndex - 1,
            beforeActioId   = $('.section').eq( ind ).attr('data-heading'),
            actionId        = $('.section').eq( nextInd ).attr('data-heading');

        
        // console.log(index-1, nextIndex-1, beforeActioId, actionId);



        if( $('.section').eq( nextInd ).hasClass('hide-overlay') || $( actionId ).hasClass('no-heading') ) {
          setTimeout(function() {
            $('.animate-text').css( 'z-index', -111 );
          }, 1300 );
          
        }else {
          $('.animate-text').css( 'z-index', 111 );
        }


        if( actionId === '' ) {
          var mainSection = $('.animate-text');
          mainSection.find('.active').removeClass('.active');
        }


        setTimeout(function() {
          $( beforeActioId ).closest('.container').removeClass('active');
        }, 1500 );
        $( beforeActioId ).removeClass('active');

        $( actionId ).addClass('active');
        $( actionId ).closest('.container').addClass('active');

      }


    });
  }


  function portfolioAnimationTextHeight() {
    $('.section').each(function() {

      var actionDiv   = $(this),
          actionId    = actionDiv.attr('data-heading'),
          height      = actionDiv.find('.inner-container').height();

      $( actionId ).css({
        'height': height
      })

    });
  }


  $(window).on('resize', function() {

    deviceControll();
    portfolioAnimationTextHeight();

  });



  $(window).on('load', function() {

    portfolioAnimationTextHeight();
    $('.home-section').addClass('active-animation');

  });



  $(document).on('ready', function() {


    deviceControll();
    fullpageInitialization( true );

    // var navigationClick = '#navbar-nav, #fp-nav';
    // $( navigationClick ).find('a').on('click', function() {
    //   $(this).closest( navigationClick ).find('.active').removeClass('active');
    //   $(this).addClass('active');
    // });



    /**
     * =============================================
     * Preloader INIT
     * =============================================
     */
    $('body').jpreLoader({
        preMainSection:     '#main-preloader',
        prePerText:         '.preloader-percentage-text',
        preBar:             '.preloader-bar',
    });




    /**
     * =============================================
     * Main Navigarion Button Script
     * =============================================
     */
    $('.nav-trigger').on('click', function() {

      var thisSection = $(this),
          actionId    = $( thisSection.attr('data-target') );

      if( thisSection.hasClass('nav-visible') ) {
        thisSection.removeClass('nav-visible')
        actionId.removeClass('show-nav');

        var setTime = setTimeout(function() {
          actionId.fadeOut();
        }, 500 );

      }else {

        actionId.fadeIn(10);
        thisSection.addClass('nav-visible')
        actionId.addClass('show-nav');
        
      }
    });



    var navigationItemHover = $('#fp-nav').find('li');
    navigationItemHover.on('mouseenter', function() {
      var actionItem = $(this).find('.fp-tooltip');
      actionItem.fadeIn(100);
      $(this).addClass('show-text');
    });
    navigationItemHover.on('mouseleave', function() {
      var actionItem = $(this).find('.fp-tooltip');
      $(this).removeClass('show-text');

      setTimeout(function() {
        actionItem.fadeOut(10);
      }, 600);
    });




    /**
     * =============================================
     * Animated Button HTML Import Script
     * =============================================
     */
    var btnMask = $('.btn-mask, .btn-nav');
    btnMask.each(function() {

      $(this).append('<span class="view-all-link-mask"><span class="view-all-link-mask-text">' + $(this).html() + '</span></span>');

    });




    /**
     * =============================================
     * Services Detail Show 
     * =============================================
     */
    $('.services-full-view').find('.services-details').fadeOut(10);

    var servicesButton = $('.btn-services');
    servicesButton.on('click', function(el) {
      el.preventDefault();

      var actionId        = $(this).attr('href'),
          hideSection     = $(this).closest('.each-services-outer'),
          parentSection   = $(this).closest('.services-section'),
          sectionHeading  = parentSection.find('.section-header');

      hideSection.addClass('active');
      sectionHeading.addClass('active');

      setTimeout(function() {
        hideSection.fadeOut(10);
        sectionHeading.fadeOut(10);
      }, 400 );

      setTimeout(function() {

        $( actionId ).fadeIn(10);
        $( actionId ).addClass('active');

      }, 550 );

    });


    /**
     * =============================================
     * Services Detail Back 3
     * =============================================
     */
    var servicesBackButton = $('.btn-services-back');
    servicesBackButton.on('click', function(el) {
      el.preventDefault();

      var actionId        = $('.each-services-outer'),
          hideSection     = $(this).closest('.services-details'),
          parentSection   = $(this).closest('.services-section'),
          sectionHeading  = parentSection.find('.section-header');

      hideSection.removeClass('active');

      setTimeout(function() {
        hideSection.fadeOut(10);
      }, 400 );

      setTimeout(function() {

        $( actionId ).fadeIn(10);
        $( actionId ).removeClass('active');

        sectionHeading.fadeIn(10);
        sectionHeading.removeClass('active');

      }, 550 );

    });




    /**
     * =============================================
     * Project Slider and Project Detail Show Button
     * =============================================
     */
    var sliderButton      = $('.btn-slider, .btn-project-detail');
    sliderButton.on('click', function(el) {
      el.preventDefault();

      var mainSectionId   = $( $(this).attr('href') ),
          dataElement     = $(this).attr('data-element'),
          dataHeading     = mainSectionId.attr('data-heading'),
          sliderId        = mainSectionId.find('.slider'),
          projectDetailId = mainSectionId.find('.project-detail'),
          imageId         = mainSectionId.find('.image');

      $( dataHeading ).addClass('no-heading');

      setTimeout(function() {
        $('.animate-text').css( 'z-index', -111 );

        if( dataElement === 'slider' ){
          sliderId.closest('.slider-outer').css( 'z-index', 111 );
          sliderId.addClass('active-section');
        }else if( dataElement === 'project-detail' ){
          projectDetailId.closest('.project-detail-outer').css( 'z-index', 111 );
          projectDetailId.addClass('active-section');
        }

      }, 600 );

      imageId.fadeOut(600);
    });


    /**
     * =============================================
     * Project Slider and Project Detail Back Button
     * =============================================
     */
    var sliderBackButton = $('.btn-slider-back, btn-project-detail-back');
    sliderBackButton.on('click', function(el) {
      el.preventDefault();

      var mainSectionId   = $( $(this).closest('.section-item') ),
          dataHeading     = mainSectionId.attr('data-heading'),
          sliderId        = mainSectionId.find('.slider'),
          projectDetailId = mainSectionId.find('.project-detail'),
          imageId         = mainSectionId.find('.image');

      imageId.fadeIn(600);
      sliderId.removeClass('active-section');
      projectDetailId.removeClass('active-section');
      $('.animate-text').css( 'z-index', 111 );
      
      setTimeout(function() {
        $( dataHeading ).removeClass('no-heading');
        sliderId.closest('.slider-outer').css( 'z-index', -111 );
        projectDetailId.closest('.project-detail-outer').css( 'z-index', -111 );
      }, 200 );

      
    });




    /**
     * =======================================
     * Image Slider
     * =======================================
     */
    var imageSlider = $(".project-slider"); // client's message
    imageSlider.owlCarousel({
      singleItem :        true,
      slideSpeed :        500,
      paginationSpeed :   500,
      autoHeight :        false,
      navigation:         true,
      pagination:         true,
      transitionStyle:    "fade"
    });




    /**
     * =======================================
     * TESTIMONIAL SYNC WITH CLIENTS
     * =======================================
     */
    var testimonialSlider = $(".testimonial-wrapper"); // client's message
    testimonialSlider.owlCarousel({
      singleItem :        true,
      autoPlay :          3000,
      slideSpeed :        500,
      paginationSpeed :   500,
      autoHeight :        false,
      navigation:         false,
      pagination:         true,
      // transitionStyle:    "fade"
    });





    /**
     * =======================================
     * Contact Form Style
     * =======================================
     */
    $( '.form-control' ).each( function( inputEl ) {

      // in case the input is already filled..
      if( $(this).val() !== '' ) {
        $(this).closest('.input-outer').addClass('input-filled');
      }

      // events:
      $( this ).focus(function(){

        $(this).closest('.input-outer').addClass('input-filled');

      });

      $( this ).blur(function(){

          if( $(this).val() === '' ) {
            $(this).closest('.input-outer').removeClass('input-filled');
          }

      });

    });





    /**
     * ============================
     * CONTACT FORM 2
     * ============================
    */
    $("#contact-form").on('submit', function(e) {
      e.preventDefault();
      var success = $(this).find('.email-success'),
        failed = $(this).find('.email-failed'),
        loader = $(this).find('.email-loading'),
        postUrl = $(this).attr('action');

      var data = {
        name: $(this).find('.contact-name').val(),
        email: $(this).find('.contact-email').val(),
        company: $(this).find('.contact-company').val(),
        subject: $(this).find('.contact-subject').val(),
        message: $(this).find('.contact-message').val()
      };

      if ( isValidEmail(data['email']) && (data['message'].length > 1) && (data['name'].length > 1) ) {
        $.ajax({
          type: "POST",
          url: postUrl,
          data: data,
          beforeSend: function() {
            loader.fadeIn(1000);
          },
          success: function(data) {
            loader.fadeOut(1000);
            success.delay(500).fadeIn(1000);
            failed.fadeOut(500);
          },
          error: function(xhr) { // if error occured
            loader.fadeOut(1000);
            failed.delay(500).fadeIn(1000);
            success.fadeOut(500);
          },
          complete: function() {
            loader.fadeOut(1000);
          }
        });
      } else {
        loader.fadeOut(1000);
        failed.delay(500).fadeIn(1000);
        success.fadeOut(500);
      }

      return false;
    });



  });


} (jQuery) );


