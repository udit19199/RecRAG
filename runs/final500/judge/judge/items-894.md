Prompt version 2026-11-10.haiku.2

You check whether retrieved text states the facts of gold evidence sentences.

Use only the retrieved items. Do not use outside knowledge and do not reason
your way to a fact the text does not state; plain rewording, synonyms,
abbreviations and different formatting are fine.

For every fact give one status:
- stated: some item says it, with every specific matching (names, numbers,
  dates, places, who did what, qualifiers).
- contradicted: some item gives a different value for the same specific, such
  as another year or another person.
- missing: neither of the above.
If one item states the fact and another contradicts it, the status is contradicted.

Evidence for stated or contradicted facts: copy one continuous passage from
one item, character for character, with no ellipses or edits, and give that
item's number. Never join separate words into a quote that does not appear as
written; if the words are apart, quote the longer continuous passage that
contains them. A fact spread across items may take several quotes. The quotes
together must contain the words that state every specific of the fact (who,
what, numbers, dates, reasons); a quote that only sets the topic, or stops
before the specific, is not evidence. Missing facts take no evidence. Every quote is checked against the item text by a
program, and a quote that does not appear there counts as missing.

Write the verdict file named in your task as a JSON array with one object per
item, in any order:
{"id": "<item id>", "facts": [{"fact": "S.F", "status": "stated", "evidence": [{"item": 3, "quote": "..."}]}, ...]}
where "S.F" is the fact's label as given (sentence number, fact number).
Give exactly one entry for every fact label of the item. Use only ids from
this file and do not skip any item.

This file has 6 items.

=== ITEM id=922da94362c5783f ===
Gold sentences and facts:
Sentence 0: In 2007 , Adobe 's first version release was Adobe Flash CS3 Professional , the ninth major version of Flash . It introduced the ActionScript 3.0 programming language , which supported modern programming practices and enabled business applications to be developed with Flash . Adobe Flex Builder ( built on Eclipse ) targeted the enterprise application development market , and was also released the same year . Flex Builder included the Flex SDK , a set of components that included charting , advanced UI , and data services ( Flex Data Services ) .
  Fact 0.0: In 2007, Adobe's first version release was Adobe Flash CS3 Professional.
  Fact 0.1: Adobe Flash CS3 Professional was the ninth major version of Flash.
  Fact 0.2: Adobe Flash CS3 Professional introduced the ActionScript 3.0 programming language.
  Fact 0.3: ActionScript 3.0 supported modern programming practices.
  Fact 0.4: ActionScript 3.0 enabled business applications to be developed with Flash.
  Fact 0.5: Adobe Flex Builder was built on Eclipse.
  Fact 0.6: Adobe Flex Builder targeted the enterprise application development market.
  Fact 0.7: Adobe Flex Builder was released in 2007.
  Fact 0.8: Flex Builder included the Flex SDK.
  Fact 0.9: The Flex SDK is a set of components that included charting, advanced UI, and data services.
  Fact 0.10: The data services in the Flex SDK are called Flex Data Services.

Retrieved items:
<item 1>
Adobe Flash
Artists may produce Flash graphics and animations using Adobe Animate . Software developers may produce applications and video games using Adobe Flash Builder ,
FlashDevelop , Flash Catalyst , or any text editor when used with the Apache Flex SDK . End - users can view Flash content via Flash Player ( for web browsers )
, AIR ( for desktop or mobile apps ) or third - party players such as Scaleform ( for video games ) . Adobe Flash Player ( supported on Microsoft Windows ,
macOS and Linux ) enables end - users to view Flash content using web browsers . Adobe Flash Lite enabled viewing Flash content on older smartphones , but has
been discontinued and superseded by Adobe AIR . The ActionScript programming language allows the development of interactive animations , video games , web
applications , desktop applications and mobile applications . Programmers can implement Flash software using an IDE such as Adobe Animate , Adobe Flash Builder
, Adobe Director , FlashDevelop and Powerflasher FDT . Adobe AIR enables full - featured desktop and mobile applications to be developed with Flash and
published for Windows , macOS , Android , iOS , Xbox One , PlayStation 4 , and Nintendo Wii U and Switch .
</item 1>

<item 2>
Adobe Flash
Previously , developers could not use the specification for making SWF - compatible players , but only for making SWF - exporting authoring software . The
specification still omits information on codecs such as Sorenson Spark , however . Animation tools ( edit ) Official tools ( edit ) Main article : Adobe Animate
The Adobe Animate authoring program is primarily used to design graphics and animation and publish the same for websites , web applications , and video games .
The program also offers limited support for audio and video embedding , and ActionScript scripting . Adobe released Adobe LiveMotion , designed to create
interactive animation content and export it to a variety of formats , including SWF . LiveMotion failed to gain any notable user base . In February 2003 ,
Macromedia purchased Presedia , which had developed a Flash authoring tool that automatically converted PowerPoint files into Flash . Macromedia subsequently
released the new product as Breeze , which included many new enhancements .
</item 2>

<item 3>
Adobe Flash
Macromedia subsequently released the new product as Breeze , which included many new enhancements . Third - party tools ( edit ) Main article : Comparison of
vector graphics editors Various free and commercial software packages can output animations into the Flash SWF format , suitable for display on the web ,
including : SWiSH Max is an animation editor with preset animation , developed by an ex-employee of Macromedia , that can output Flash animations Toon Boom is a
traditional animation tool that can output Flash animations Apple Keynote allows users to export presentations to Flash SWF animations Xara Photo & Graphic
Designer can output Flash animations Ajax Animator aim to create a Flash development environment KToon can edit vectors and generate SWF , but its interface is
very different from Macromedia 's Screencast and Screencam , produces demos or tutorials by capturing the screen and generating a Flash animation of the same
GoAnimate is a software as a service tool to create animated videos .
</item 3>

<item 4>
Adobe Flash
Adobe Wallaby Server - side Adobe Flash Media Server Obsolete software Adobe Flash Builder Adobe Flash Catalyst Adobe Flash Lite SWFAddress SWFObject MTASC
Scaleform GFx SWiSH Max Swift 3D Papervision3D Ajax Animator Shumway Swfdec OpenLaszlo Related topics Comparison of HTML5 and Flash Apple and Adobe Flash
controversy Flash animation films television Protected Streaming Real - Time Messaging Protocol Real - Time Media Flow Protocol Developers FutureWave Software
Macromedia Adobe Systems Adobe Creative Suite and Creative Cloud Current products Acrobat After Effects Animate Audition Bridge Dreamweaver Flash Builder
Illustrator InCopy InDesign Lightroom Media Encoder Muse Photoshop Prelude Premiere Pro Spark XD Creative Cloud services Behance Business Catalyst Story Typekit
Former products Contribute Device Central Edge Edge Animate Encore Flash Catalyst Fireworks GoLive ImageReady OnLocation Soundbooth Stock Photos Ultra Adobe
eLearning Suite Current products Captivate Flash Dreamweaver Photoshop Acrobat Presenter Soundbooth Shared applications Bridge Device Central Adobe Systems
Software Suites Creative Cloud ( Creative Suite ) eLearning Suite Technical Communication Suite Desktop Animate Acrobat After Effects Brackets Edge Code Edge
Reflow Digital Editions Director Dreamweaver FrameMaker FreeHand Flash Player Flex Builder GoLive Illustrator InDesign Media Encoder Media Player Muse PageMaker
Photoshop
</item 4>

<item 5>
Adobe Flash
FreeHand Flash Player Flex Builder GoLive Illustrator InDesign Media Encoder Media Player Muse PageMaker Photoshop Photoshop Lightroom Premiere Pro Reader
Shockwave Player Server LiveCycle Flash Media Server BlazeDS ColdFusion JRun PhoneGap Build Technologies ActionScript Adobe AIR Adobe Flash Adobe Flex Adobe
Shockwave Adobe Font Folio Digital Negative ( DNG ) Authorware FlashPaper Portable Document Format ( PDF ) PostScript Web services Acrobat.com Adobe Solutions
Network Adobe Photoshop Express Adobe Premiere Express Adobe Spark Adobe Stock Acquisitions Aldus Macromedia Scene7 Omniture Behance Fotolia People Bruce Chizen
Charles Geschke Shantanu Narayen John Warnock Del Yocam Commons Retrieved from `` https://en.wikipedia.org/w/index.php?title=Adobe_Flash&oldid=821675124 ''
Categories : Adobe Flash American inventions Cross-platform software Computing platforms C++ software Mac OS multimedia software MacOS multimedia software
Windows multimedia software Macromedia software Adobe software 1993 software Video game development software Hidden categories : Webarchive template wayback
links Pages using web citations with no URL Pages using citations with accessdate and no URL Use mdy dates from April 2012 Articles containing potentially dated
statements from February 2009 All articles containing potentially dated statements Articles containing potentially
</item 5>

<item 6>
Adobe Flash
, blend modes ( similar to Adobe Photoshop ) , and advanced features for FLV video . Adobe ( edit ) Macromedia was acquired by Adobe Systems in 2005 , and the
entire Macromedia product line including Flash , Dreamweaver , Director / Shockwave , Fireworks ( which has since been discontinued ) and Authorware is now
handled by Adobe . In 2007 , Adobe 's first version release was Adobe Flash CS3 Professional , the ninth major version of Flash . It introduced the ActionScript
3.0 programming language , which supported modern programming practices and enabled business applications to be developed with Flash . Adobe Flex Builder (
built on Eclipse ) targeted the enterprise application development market , and was also released the same year . Flex Builder included the Flex SDK , a set of
components that included charting , advanced UI , and data services ( Flex Data Services ) . In 2008 , Adobe released the tenth version of Flash , Adobe Flash
CS4 . Flash 10 improved animation capabilities within the Flash editor , adding a motion editor panel ( similar to Adobe After Effects ) , inverse kinematics (
bones ) , basic 3D object animation , object - based animation , and other text and graphics features .
</item 6>

<item 7>
Adobe Flash Player
( SWC ) Software ( list ) Players Adobe AIR Adobe Flash Player OpenFL Gnash Lightspark Gameswf Tamarin Animation tools Adobe Animate Toon Boom Moho GoAnimate
Stencyl Programming tools FlashDevelop Powerflasher FDT Apache Flex Haxe CrossBridge Adobe Scout FlashFirebug Libraries Starling Framework Away3D Flare3D
Stage3D PureMVC Ming List of Flex frameworks Converters Adobe Flash Media Live Encoder SWFTools swfmill Google Swiffy Adobe Wallaby Server - side Adobe Flash
Media Server Obsolete software Adobe Flash Builder Adobe Flash Catalyst Adobe Flash Lite SWFAddress SWFObject MTASC Scaleform GFx SWiSH Max Swift 3D
Papervision3D Ajax Animator Shumway Swfdec OpenLaszlo Related topics Comparison of HTML5 and Flash Apple and Adobe Flash controversy Flash animation films
television Protected Streaming Real - Time Messaging Protocol Real - Time Media Flow Protocol Developers FutureWave Software Macromedia Adobe Systems Adobe
Systems Software Suites Creative Cloud ( Creative Suite ) eLearning Suite Technical Communication Suite Desktop Animate Acrobat After Effects Brackets Edge Code
Edge Reflow Digital Editions Director Dreamweaver FrameMaker FreeHand Flash Player Flex Builder GoLive Illustrator InDesign Media Encoder Media Player Muse
</item 7>

<item 8>
Adobe Flash
Adobe Flash - Wikipedia Adobe Flash Jump to : navigation , search For the application / animation - maker , see Adobe Animate . For other uses , see Adobe Flash
( disambiguation ) . Adobe Flash Developer ( s ) Adobe Systems Macromedia FutureWave Target platform ( s ) Web browsers , iOS , Android , Windows , macOS Editor
software Adobe Animate , Flash Builder , FlashDevelop , Powerflasher FDT , Flash Catalyst Scaleform Player software Flash Player Adobe AIR OpenFL Scaleform
Gnash Lightspark Format ( s ) SWF FLV FLA Programming language ( s ) ActionScript Application ( s ) Animation Browser games Rich Internet applications Desktop
applications Mobile apps Mobile games console games computer games Status Active ( Until 2020 ) License Proprietary Adobe Flash is a multimedia software
platform used for production of animations , rich Internet applications , desktop applications , mobile applications , mobile games and embedded web browser
video players . Flash displays text , vector graphics and raster graphics to provide animations , video games and applications . It allows streaming of audio
and video , and can capture mouse , keyboard , microphone and camera input . Artists may produce Flash graphics and animations using Adobe Animate .
</item 8>

<item 9>
Adobe Flash
The Apache Flex ActionScript 3.0 compiler generates SWF files from ActionScript 3 files . Flex was the primary ActionScript 3 compiler and was actively
developed by Adobe before it was donated to Apache Software Foundation in 2011 . Adobe Animate -- primarily used to design graphics and animation , but supports
ActionScript scripting and debugging . Adobe Flash Builder -- enterprise application development & debugging , contains the Flex SDK with UI and charting
components . Adobe Scout -- a visual profiler to optimize performance of Flash content . CrossBridge -- a free SDK to cross-compile C++ code to run in Flash
Player . Third - party tools ( edit ) Third - party development tools have been created to assist developers in creating software applications and video games
with Flash . FlashDevelop is a free and open - source Flash ActionScript IDE , which includes a project manager and debugger for building applications on Flash
Player and Adobe AIR . Powerflasher FDT is a commercial ActionScript IDE similar to FlashDevelop . Haxe is an open source , high - level object - oriented
programming language geared towards web - content creation that can compile SWF files from Haxe programs .
</item 9>

<item 10>
Adobe Flash
CreateJS is library that while available separately was also adopted by Adobe as a replacement for Wallaby in CS6 . Unlike Wallaby , which was a standalone
program , the `` Toolkit for CreateJS '' only works as a plug - in inside Flash Professional ; it generates output for the HTML5 canvas , animated with
JavaScript . Around December 2013 , the toolkit was integrated directly into Flash Professional CC . Criticisms ( edit ) Mobile support ( edit ) Websites built
with Adobe Flash will not function on most modern mobile devices running Google Android or iOS ( iPhone , iPad ) . The only alternative is using HTML5 and
responsive web design to build websites that support both desktop and mobile devices . However , Flash is still used to build mobile games using Adobe AIR .
Such games will not work in mobile web browsers , but must be installed via the appropriate app store . Vendor dependence ( edit ) See also : Vendor lock - in
The reliance on Adobe for decoding Flash makes its use on the World Wide Web a concern -- the completeness of its public specifications are debated , and no
complete implementation of Flash is publicly available in source code form with a license that permits reuse .
</item 10>
=== END ITEM id=922da94362c5783f ===

=== ITEM id=0abb0fa1c647a0d1 ===
Gold sentences and facts:
Sentence 0: `` Sacrifice '' is a ballad performed by musician Elton John . The lyrics are by Bernie Taupin and the music by John . The song appears on the 1989 album Sleeping with the Past . It was first released in October 1989 , then in 1990 , and was the second single from the album . It achieved success , particularly in France and the UK , where it became his first solo chart - topper , spending five weeks at the top .
  Fact 0.0: 'Sacrifice' is a ballad performed by musician Elton John.
  Fact 0.1: The lyrics of 'Sacrifice' are by Bernie Taupin.
  Fact 0.2: The music of 'Sacrifice' is by Elton John.
  Fact 0.3: 'Sacrifice' appears on the 1989 album Sleeping with the Past.
  Fact 0.4: 'Sacrifice' was first released in October 1989.
  Fact 0.5: 'Sacrifice' was released in 1990.
  Fact 0.6: 'Sacrifice' was the second single from the album Sleeping with the Past.
  Fact 0.7: 'Sacrifice' achieved success, particularly in France and the UK.
  Fact 0.8: 'Sacrifice' became Elton John's first solo chart-topper in France and the UK.
  Fact 0.9: 'Sacrifice' spent five weeks at the top of the chart.

Retrieved items:
<item 1>
Sacrifice (song)
000 Personnel ( edit ) Elton John : vocals , Roland RD - 1000 digital piano Davey Johnstone : electric guitar Romeo Williams : bass Jonathan Moffett : drums Guy
Babylon : keyboards Fred Mandel : keyboards Preceded by `` Words '' by The Christians French SNEP number - one single 26 May 1990 -- 9 June 1990 ( 3 weeks )
Succeeded by `` Bo le lavabo ( WC Kiss ) '' by Lagaf ' Preceded by `` Vogue '' by Madonna Eurochart Hot 100 number - one single 16 June 1990 -- 28 July 1990 ( 7
weeks ) Succeeded by `` U Ca n't Touch This '' by MC Hammer Preceded by `` World In Motion '' by Englandneworder UK number - one single 17 June 1990 -- 21 July
1990 ( 5 weeks ) Succeeded by `` Turtle Power '' by Partners in Kryme References ( edit ) ^ Jump up to : `` Elton John - Full Official Chart History '' .
Official Charts Company . Official Charts Company . Retrieved 20 January 2016 .
</item 1>

<item 2>
Sacrifice (song)
Sacrifice ( song ) - wikipedia Sacrifice ( song ) Jump to : navigation , search For other songs with the same title , see Sacrifice ( disambiguation ) § Songs .
`` Sacrifice '' Single by Elton John from the album Sleeping with the Past A-side `` Healing Hands '' ( second release ) B - side `` Love Is a Cannibal '' (
first release ) Released October 1989 ( first release ) May 1990 ( second release ) Format CD , vinyl record ( 7 '' and 12 `` ) , audio cassette Length 5 : 11
Label Rocket , MCA Songwriter ( s ) Elton John , Bernie Taupin Producer ( s ) Chris Thomas Elton John singles chronology `` Healing Hands '' ( 1989 ) ``
Sacrifice '' ( 1989 ) `` Club at the End of the Street '' ( 1990 ) `` Healing Hands '' ( 1989 ) `` Sacrifice '' ( 1989 ) `` Club at the End of the Street '' (
1990 ) `` Sacrifice '' is a ballad performed by musician Elton John . The lyrics are by Bernie Taupin and the music by John .
</item 2>

<item 3>
Sacrifice (song)
Awarded as The Best Song Retrieved from `` https://en.wikipedia.org/w/index.php?title=Sacrifice_(song)&oldid=833955469 '' Categories : 1989 singles 1990 singles
Disney songs Elton John songs European Hot 100 Singles number - one singles SNEP Top Singles number - one singles UK Singles Chart number - one singles Songs
with lyrics by Bernie Taupin Songs with music by Elton John Pop ballads Rock ballads Song recordings produced by Chris Thomas ( record producer ) 1989 songs MCA
Records singles The Rocket Record Company singles Music videos directed by Alek Keshishian Hidden categories : CS1 Dutch - language sources ( nl ) CS1 French -
language sources ( fr ) Webarchive template wayback links CS1 Norwegian - language sources ( no ) CS1 German - language sources ( de ) Articles with hAudio
microformats Use dmy dates from August 2010 Talk Contents About Wikipedia Dansk Español Français Italiano Nederlands Norsk nynorsk Polski Português Edit links
This page was last edited on 3 April 2018 , at 05 : 22 .
</item 3>

<item 4>
Graph query row:
song: Sacrifice
album: Sleeping with the Past
</item 4>

<item 5>
Sacrifice (song)
Jump up ^ `` UK certifications , database '' . Bpi . Retrieved 9 September 2008 . External links ( edit ) Lyrics of this song at MetroLyrics Elton John songs
1960s and 1970s `` Skyline Pigeon '' ( 1968 ) `` I 've Been Loving You '' ( 1968 ) `` Lady Samantha '' ( 1969 ) `` It 's Me That You Need '' ( 1969 ) `` Border
Song '' ( 1970 ) `` Rock and Roll Madonna '' ( 1970 ) `` From Denver to L.A. '' ( 1970 ) `` Take Me to the Pilot '' ( 1970 ) `` Your Song '' ( 1970 ) `` Friends
'' ( 1971 ) `` Levon '' ( 1971 ) `` Tiny Dancer '' ( 1972 ) `` Rocket Man '' ( 1972 ) `` Honky Cat '' ( 1972 ) `` I Think I 'm Going to Kill Myself '' ( 1972 )
`` Mona Lisas and Mad Hatters '' ( 1972 ) `` Crocodile Rock '' ( 1972 ) `` Elderberry Wine '' (
</item 5>

<item 6>
Sacrifice (song)
The song is beloved by both John and Bernie Taupin and deemed a bookend of sorts to their first hit `` Your Song '' . Taupin has said he is surprised he wrote
it , and said in an interview , `` I think ' Sacrifice ' is one of the best songs we 've written . '' The song is not a typical love song , but rather a song
about a breakup of a marriage where the loss of the relationship is `` no sacrifice . '' Although the single was successful , Elton John did not tour Europe in
both 1989 -- 1990 to support the Sleeping with the Past album ( with the exception of one appearance in Knebworth Park on 30 June 1990 ) touring only in the
U.S. and Australia . The 1989 European tour was , instead , in support of the Reg Strikes Back album , with no material representing Sleeping with the Past . In
all shows that were done to support Sleeping with the Past in 1989 -- 1990 , Elton played this song solo on his Roland RD - 1000 Digital Piano .
</item 6>

<item 7>
Sacrifice (song)
'' ( 1997 ) `` Something About the Way You Look Tonight '' ( 1997 ) `` Candle in the Wind 1997 '' ( 1997 ) `` Written in the Stars '' ( 1999 ) 2000s `` I Want
Love '' ( 2001 ) `` This Train Do n't Stop There Anymore '' ( 2002 ) `` Original Sin '' ( 2002 ) `` Are You Ready for Love '' ( 2003 ) `` Ghetto Gospel '' (
2005 ) `` Electricity '' ( 2005 ) `` The Bridge '' ( 2006 ) `` Joseph , Better You than Me '' ( 2008 ) `` Tiny Dancer ( Hold Me Closer ) '' ( 2009 ) Related
articles Albums discography Videography Book : Elton John Category : Elton John Ivor Novello Awards -- Best Song Musically and Lyrically 1960s She 's Leaving
Home ( 1968 ) 1970s Where Do You Go To ( My Lovely ) ? ( 1970 ) Something ( 1971 ) Do n't Let It Die ( 1972 ) Without You ( 1973 ) Daniel ( 1974 ) Streets
</item 7>

<item 8>
Sacrifice (song)
Billboard Hot Adult Contemporary Tracks Chart ( 1990 ) Peak position Australian ARIA Singles Chart 7 Dutch Mega Top 100 Eurochart Hot 100 French SNEP Singles
Chart German Singles Chart 36 Irish Singles Chart New Zealand RIANZ Singles Chart 7 Norwegian Singles Chart Swiss Singles Chart 23 UK Singles Chart U.S.
Billboard Hot 100 18 Zimbabwe Singles Chart 6 Year - end charts ( edit ) Chart ( 1990 ) Rank Australian Singles Chart 47 Dutch Top 40 11 UK Singles Charts
Certifications and sales ( edit ) Country Certification Date Sales certified Physical sales France Silver 200,000 380,000 UK Platinum 1 September 1990 600,000
Personnel ( edit ) Elton John : vocals ,
</item 8>

<item 9>
Sacrifice (song)
The lyrics are by Bernie Taupin and the music by John . The song appears on the 1989 album Sleeping with the Past . It was first released in October 1989 , then
in 1990 , and was the second single from the album . It achieved success , particularly in France and the UK , where it became his first solo chart - topper ,
spending five weeks at the top . A country music cover of the song was performed by Don Henley and Vince Gill as part of the 2018 tribute album Restoration :
Reimagining the Songs of Elton John and Bernie Taupin . Contents ( hide ) 1 Overview 1.1 Background and writing 1.2 Music video 1.3 Chart performance 2 Track
listings 3 Charts and sales 3.1 Weekly charts 3.2 Year - end charts 3.3 Certifications and sales 4 Personnel 5 References 6 External links Overview ( edit )
Background and writing ( edit ) `` Sacrifice '' is inspired both by the Aretha Franklin song `` Do Right Woman , Do Right Man '' and described as a Percy Sledge
song by Elton John .
</item 9>

<item 10>
Sacrifice (song)
Thus , it became John 's first solo number - one single in the UK Singles Chart , remaining on the top spot for five weeks . With this re-release , Elton John
also got his first number one in France and stayed on the chart for 26 weeks . Track listings ( edit ) First release 12 '' single -- UK `` Sacrifice '' -- 5 :
07 `` Love Is a Cannibal '' ( From the Columbia Motion Picture Ghostbusters II ) -- 3 : 53 CD single -- U.S. `` Sacrifice '' -- 5 : 07 `` Love Is a Cannibal ''
( from the Columbia Motion Picture Ghostbusters II ) -- 3 : 53 `` Durban Deep '' Second release UK single `` Sacrifice '' -- 5 : 07 `` Healing Hands '' -- 4 :
22 Charts and sales ( edit ) Weekly charts ( edit ) Chart ( 1989 ) Peak position UK Singles Chart 55 U.S.
</item 10>
=== END ITEM id=0abb0fa1c647a0d1 ===

=== ITEM id=075ebcbd51d04017 ===
Gold sentences and facts:
Sentence 0: The first installment , Divergent ( 2014 ) , grossed over $288 million worldwide , while the second installment , The Divergent Series : Insurgent ( 2015 ) , grossed over $297 million worldwide . Insurgent was also the first Divergent film to be released in IMAX 3D . The third installment , The Divergent Series : Allegiant ( 2016 ) , grossed $179 million . Thus , the first three films of the series have grossed over $765 million worldwide . A fourth film , The Divergent Series : Ascendant was never made , due to Allegiant 's poor showing at the box office .
  Fact 0.0: Divergent (2014) is the first installment of the Divergent series.
  Fact 0.1: Divergent grossed over $288 million worldwide.
  Fact 0.2: The Divergent Series: Insurgent (2015) is the second installment of the Divergent series.
  Fact 0.3: The Divergent Series: Insurgent grossed over $297 million worldwide.
  Fact 0.4: The Divergent Series: Insurgent was the first Divergent film released in IMAX 3D.
  Fact 0.5: The Divergent Series: Allegiant (2016) is the third installment of the Divergent series.
  Fact 0.6: The Divergent Series: Allegiant grossed $179 million.
  Fact 0.7: The first three films of the Divergent series have grossed over $765 million worldwide in total.
  Fact 0.8: The Divergent Series: Ascendant is a fourth film that was never made.
  Fact 0.9: The Divergent Series: Ascendant was never made due to The Divergent Series: Allegiant's poor showing at the box office.

Retrieved items:
<item 1>
The Divergent Series is a dystopian science-fiction film trilogy based on Veronica Roth’s novels, starring Shailene Woodley, Theo James, Zoë Kravitz, Miles
Teller, and Ansel Elgort. Summit Entertainment and Lionsgate released Divergent (2014), Insurgent (2015), and Allegiant (2016); a planned fourth film was
cancelled. The three films cost $305 million combined and grossed about $765 million, but reviews declined with
</item 1>

<item 2>
The Divergent Series
For the mathematical use , see Divergent series . The Divergent Series Produced by Lucy Fisher Douglas Wick Pouya Shabazian Based on The Divergent trilogy by
Veronica Roth Starring Shailene Woodley Theo James Zoë Kravitz Miles Teller Ansel Elgort Production company Red Wagon Entertainment Summit Entertainment
Distributed by Summit Entertainment Lionsgate Release date 1 : March 21 , 2014 ( 2014 - 03 - 21 ) 2 : 000000002015 - 03 - 20 - 0000 March 20 , 2015 3 :
000000002016 - 03 - 18 - 0000 March 18 , 2016 4 : Cancelled Running time 379 minutes ( 3 films ) Country United States Language English Budget Total ( 3 films )
: $305 million Box office Total ( 3 films ) : $765,409,015 The Divergent Series is a feature film trilogy based on the Divergent novels by the American author
Veronica Roth . Distributed by Summit Entertainment and Lionsgate Films , the series consists of three science fiction films set in a dystopian society .
</item 2>

<item 3>
The Divergent Series
On December 16 , 2013 , it was announced that Neil Burger , director of Divergent , would not return to direct Insurgent as he was still working on the first
film . On February 13 , 2014 , it was announced that Robert Schwentke was offered the director position for the film and that Akiva Goldsman had been hired to
re-write Duffield 's script . Also in December 2013 , Summit Entertainment announced that the film adaptation of Allegiant , the third and final novel in the
Divergent trilogy , would be released in March 2016 , serving as the finale of the series , which at the time was planned as a trilogy , but on April 11 , 2014
, the studio decided to split the novel into a two - part film , much the like the Harry Potter , Twilight , and Hunger Games franchises did with the finales of
their series .
</item 3>

<item 4>
The *Divergent* film series adapted Veronica Roth’s novels, starring Shailene Woodley as Tris and Theo James as Four, alongside Ansel Elgort, Zoë Kravitz, Miles
Teller, and Kate Winslet. *Divergent* (2014), *Insurgent* (2015), and *Allegiant* (2016) were directed by Neil Burger and Robert Schwentke and grossed about
$765 million worldwide. Reviews weakened across
</item 4>

<item 5>
The Divergent film series adapts Veronica Roth’s trilogy. Produced by Lucy Fisher, Pouya Shabazian, and Douglas Wick, it stars Shailene Woodley as Tris and Theo
James as Four, with Ansel Elgort, Zoë Kravitz, Miles Teller, and Kate Winslet among the cast. Neil Burger directed Divergent (2014); Robert Schwentke directed
Insurgent (2015) and Allegiant (2016). Development began
</item 5>

<item 6>
The Divergent Series
by Veronica Roth Composer Junkie XL Joseph Trapanese N / A Cinematographer Alwin H. Küchler Florian Ballhaus Editor ( s ) Richard Francis - Bruce Nancy
Richardson Nancy Richardson Stuart Levy Stuart Levy N / A Production companies Red Wagon Entertainment Summit Entertainment Red Wagon Entertainment Summit
Entertainment Mandeville Films Red Wagon Entertainment Mandeville Films N / A Distributor Lionsgate Summit Entertainment N / A Running time 139 minutes 119
minutes 121 minutes TBA Release date March 21 , 2014 March 20 , 2015 March 18 , 2016 TBA Reception ( edit ) Further information on the reception of each film :
Divergent ( film ) § Reception , The Divergent Series : Insurgent § Reception , and The Divergent Series : Allegiant § Reception Though the first two films were
commercially successful , the series as a whole has not been well received by critics . Each successive film release has been rated lower than the previous
release on review aggregator websites . Negative reviews of the third novel Allegiant and the studios ' decision to split it into two films have been blamed for
the third film 's lackluster performance at the box office .
</item 6>

<item 7>
The Divergent Series
The first installment , Divergent ( 2014 ) , grossed over $288 million worldwide , while the second installment , The Divergent Series : Insurgent ( 2015 ) ,
grossed over $297 million worldwide . Insurgent was also the first Divergent film to be released in IMAX 3D . The third installment , The Divergent Series :
Allegiant ( 2016 ) , grossed $179 million . Thus , the first three films of the series have grossed over $765 million worldwide . A fourth film , The Divergent
Series : Ascendant was never made , due to Allegiant 's poor showing at the box office .
</item 7>

<item 8>
A fourth Divergent film, adapting the second half of Allegiant, was initially planned for theaters in March 2017, later moved to June, with Lee Toland Krieger
attached to direct. After the third film’s poor box-office performance, producers considered making it a TV movie that could introduce characters for a spinoff
series beyond the books. Shailene Woodley said the format was undecided and was doubtful she would return for TV. Once the TV project was confirmed in February
</item 8>

<item 9>
The Divergent Series
The Divergent Series : Allegiant ( 2016 ) ( edit ) Main article : The Divergent Series : Allegiant On December 16 , 2013 , Summit Entertainment announced that
the film adaptation of Allegiant would be released on March 18 , 2016 . On April 11 , 2014 , Lionsgate announced that the film adaptation would be split into
two films with the first part titled , The Divergent Series : Allegiant -- Part 1 . The film was then re-titled as The Divergent Series : Allegiant . Tris must
escape with Four and go beyond the wall enclosing Chicago . For the first time ever , they will leave the only city and family they have ever known in order to
find a peaceful solution for their embroiled city . Once outside , old discoveries are quickly rendered meaningless with the revelation of shocking new truths .
Tris must quickly decide who she can trust as a ruthless battle ignites beyond the walls of Chicago . In order to survive , Tris will be forced to make
impossible choices about courage , allegiance , sacrifice and love .
</item 9>

<item 10>
The Divergent Series
I 'm thrilled ! '' The producers said about his casting : `` We took our time to find the right actor to fill the role of Four , and Theo is definitely the
perfect fit . Veronica has crafted a truly iconic character in Four and we can not wait to begin production and bring him and this story to life for millions of
fans around the world . '' Production ( edit ) Filming for the franchise began on April 16 , 2013 . Film Producers Director Screenwriters Novel by Veronica Roth
Divergent Fisher , Lucy Lucy Fisher Douglas Wick Pouya Shabazian Burger , Neil Neil Burger Evan Daugherty Vanessa Taylor Divergent The Divergent Series :
Insurgent Schwentke , Robert Robert Schwentke Akiva Goldsman Brian Duffield Insurgent The Divergent Series : Allegiant Noah Oppenheim Adam Cooper Bill Collage
Allegiant The Divergent Series : Ascendant Lee Toland Krieger Lucy Fisher , Douglas Wick , and Pouya Shabazian have been producers for all four films in the
series . Veronica Roth and Neil Burger have acted as executive producers on the first two films .
</item 10>
=== END ITEM id=075ebcbd51d04017 ===

=== ITEM id=8943781ef33850e8 ===
Gold sentences and facts:
Sentence 0: In 1918 , Firestone Tire and Rubber Company of Canada was incorporated in Hamilton , Ontario and in 1922 , the first Canadian - made tire rolled off the line on September 15 . During the 1920s , Firestone produced the Oldfield tire , named for racing driver Barney Oldfield .
  Fact 0.0: Firestone Tire and Rubber Company of Canada was incorporated in 1918.
  Fact 0.1: Firestone Tire and Rubber Company of Canada was incorporated in Hamilton, Ontario.
  Fact 0.2: In 1922, the first Canadian-made tire rolled off the line on September 15.
  Fact 0.3: During the 1920s, Firestone produced the Oldfield tire.
  Fact 0.4: The Oldfield tire was named for racing driver Barney Oldfield.
Sentence 1: Bridgestone / Firestone also produces tires under the Fuzion brand name . Fuzion is a tire manufactured in China for Bridgestone / Firestone as a budget replacement tire .
  Fact 1.0: Bridgestone / Firestone also produces tires under the Fuzion brand name.
  Fact 1.1: Fuzion is a tire manufactured in China for Bridgestone / Firestone.
  Fact 1.2: Fuzion is a budget replacement tire.

Retrieved items:
<item 1>
Firestone Tire and Rubber Company
Retrieved 2007 - 03 - 21 . Jump up ^ Isidore , Chris ( May 21 , 2001 ) . `` Firestone severs ties to Ford '' . CNN . Jump up ^ NHTSA Recall Campaign 01T - 016
Q4 Vehicle Safety Recall Quarterly Report Information , January 25 , 2008 ( Amended March 26 , 2009 ) ( from the National Highway Traffic Safety Administration
) ^ Jump up to : Channel 2 Investigation Finds Recalled Tires for Sale , November 4 , 2013 Archived November 5 , 2013 , at the Wayback Machine . Jump up ^ ``
Firestone Jingle '' ( QuickTime ) . Firestone . Retrieved 2007 - 03 - 22 . External links ( edit ) Wikimedia Commons has media related to Firestone Tire and
Rubber Company .
</item 1>

<item 2>
Firestone Tire and Rubber Company
ISBN 978 - 0 - 15 - 692620 - 1 . Jump up ^ `` A Global Dream '' . Bridgestone Americas . Retrieved 3 August 2013 . Jump up ^ `` History of Bridgestone /
Firestone '' . Bridgestone / Firestone Canada . 2009 . Archived from the original on 10 May 2010 . Jump up ^ Firestone 8000 Tire Order by Ford Ford Chooses
Firestone for Model T Jump up ^ Canada manufactures first Firestone tire Archived 2010 - 05 - 10 at the Wayback Machine . Firestone Canada Incorporated Jump up
^ Chris Nichols ( 2013 - 03 - 01 ) . `` DispL. A. Case # 52 : L.A. 's First Firestone Tire - Los Angeles Magazine '' . Lamag.com . Retrieved 2015 - 12 - 17 .
Jump up ^ Lost Memphis Firestone Tire & Rubber Company Jump up ^ Firestone Closings Jump up ^ Peck , Merton J.
</item 2>

<item 3>
Firestone Tire and Rubber Company
The combined Bridgestone / Firestone North American operations are now based in Nashville , Tennessee . In 2008 , the companies celebrated a 20 - year
anniversary of the merger , and changed the tire division name to Bridgestone Americas Tire Operations , LLC . In 2012 , Bridgestone Americas opened a $100
million technical center in Akron . Firestone tires used for the Indianapolis 500 race . Apart from tires , several companies and divisions operate with the
Firestone brand in its name . These companies include Firestone Building Products , LLC , Firestone Industrial Products , LLC , Firestone Complete Auto Care ,
Firestone Natural Rubber Company and Firestone Specialty Products . Firestone Building Products and Firestone Industrial Products are headquartered in
Indianapolis , Indiana and operate in 23 U.S. states and ten countries , with 11,000 employees worldwide . The company reports annual sales of more than $2.5
billion . In 1972 , Firestone received a ten - year import `` concession '' by the Kenyan government to secure Firestone 's investment in a domestic tire plant
, which gave it a virtual monopoly . This included both general price and foreign exchange controls .
</item 3>

<item 4>
Firestone Tire and Rubber Company
External links ( edit ) Wikimedia Commons has media related to Firestone Tire and Rubber Company . Official website Century of Progress from the Summit Memory
Project Bridgestone Corporation Subsidiaries Firestone Diversified Products Firestone Tire and Rubber Company Firestone - Apsley Rubber Company Products
Bridgestone Picnica Helmz Facilities Firestone Tyre Factory People Harvey Samuel Firestone Related Akron Firestone Non-Skids Bridgestone Arena Bridgestone
Museum of Art Bridgestone Open Firestone and Ford tire controversy Firestone Country Club International Labor Rights Forum v Firestone Tire and Rubber Co The
Voice of Firestone WGC - Bridgestone Invitational Automotive industry in the United States Automotive industry Economy of the United States Transportation in
the United States American vehicle manufacturers ( list ) AGCO Challenger Tractor Massey Ferguson AM General American Expedition Vehicles American Growler Amp
Electric Vehicles Anteros Coachworks Arcimoto Armour Group , Inc .
</item 4>

<item 5>
Firestone Tire and Rubber Company
Founded on August 3 , 1900 , the company initiated operations with 12 employees . Together , Firestone and Goodyear were the largest suppliers of automotive
tires in North America for over 75 years . In 1906 Henry Ford chose Firestone for Model T original equipment tires . In 1918 , Firestone Tire and Rubber Company
of Canada was incorporated in Hamilton , Ontario and in 1922 , the first Canadian - made tire rolled off the line on September 15 . During the 1920s , Firestone
produced the Oldfield tire , named for racing driver Barney Oldfield . In 1926 , the company opened one of the world 's biggest rubber plantations in Liberia ,
West Africa , spanning more than 1 million acres . 1926 was also the year that the company opened its first Firestone Complete Auto Care store ( Firestone
Complete Auto Care is a division of Firestone that offers automotive maintenance and repair ) . In 1927 , Henry Ford and tire maven Harvey Firestone took a trip
to Los Angeles to select locations for their new factories .
</item 5>

<item 6>
Firestone Tire and Rubber Company
Firestone tire and Rubber company - wikipedia Firestone tire and Rubber company Jump to : navigation , search Firestone Tire and Rubber Company Type Subsidiary
Industry Manufacturing Founded Akron , Ohio , United States ( August 3 , 1900 ; 117 years ago ( 1900 - 08 - 03 ) ) Founder Harvey Firestone Headquarters 200 4th
Ave S , Nashville , Tennessee , United States Area served Worldwide Key people Gordon Knapp ( CEO and President ) Bill Thompson ( COO ) Products Tires Revenue
US $ 2.09 billion ( 2004 ) Number of employees 40,000 Parent Bridgestone ( 1988 -- present ) Website firestone.com The Firestone Tire and Rubber Company is an
American tire company founded by Harvey Samuel Firestone in 1900 to supply pneumatic tires for wagons , buggies , and other forms of wheeled transportation
common in the era . Firestone soon saw the huge potential for marketing tires for automobiles . The company was a pioneer in the mass production of tires .
Harvey Firestone had a friendship with Henry Ford .
</item 6>

<item 7>
Firestone Tire and Rubber Company
Willys Marathon Motor Works Marmon Roosevelt Marvel Motors Matbro Mercer Monaco Coach Muntz Car Company Oliver Farm Equipment Packard Peerless Motor Company
Pierce - Arrow Sebring Vanguard Sterling Trucks Studebaker Erskine Rockne Stutz Twentieth Century Motor Car Corporation United Defense VL White Wildfire Defunct
factories Brampton Assembly ( AMC ) Diamond - Star Motors Fiat Chrysler factories closed Ford factories closed General Motors factories closed NUMMI Packard
Automotive Plant Volkswagen Westmoreland Assembly Related topics AAA Chicago Auto Show Interstate Highway System National Highway Traffic Safety Administration
New York International Auto Show North American International Auto Show SAE International ( 1 ) Although New Flyer is Canadian , their Subsidiaries , NABI and
Motor Coach Industries , are headquartered in the U.S. ( 2 ) Former meaning the company is no longer in the automotive manufacturing business Portal Retrieved
from `` https://en.wikipedia.org/w/index.php?title=Firestone_Tire_and_Rubber_Company&oldid=806664689 '' Categories : 1900 establishments in Ohio American
companies established in 1900 Tire manufacturers Automotive companies of the United States Companies based in Akron , Ohio Manufacturing companies based in
Nashville , Tennessee Bridgestone U.S.
</item 7>

<item 8>
Firestone Tire and Rubber Company
East Los Angeles College has proposed a new satellite campus at the site . In 1928 the company built a factory in Brentford , England , a longtime Art Deco
landmark on a major route into the city . This closed in 1979 . In 1936 the company opened a plant in Memphis , Tennessee . With a work force exceeding 3,000
employees , the Memphis plant was the largest tire manufacturer in the company 's worldwide operation . On July 1 , 1963 , the company celebrated the production
of 100 million tires in Memphis . The plant was closed in 1982 . During World War II the company was called on by the U.S. Government to make artillery shells ,
aluminum kegs for food transport and rubberized military products . Firestone ranked 55th among United States corporations in the value of World War II military
production contracts . In the 1940s , Firestone was given a defense contract to produce plastic helmet liners . While outproduced by Westinghouse Electric they
still made a fair amount for the M1 Helmet .
</item 8>

<item 9>
Firestone Tire and Rubber Company
Friends say Ford wanted to be near the ocean and picked Long Beach and suggested Firestone go to South Gate , California . The tiny community southeast of
Downtown was mostly agriculture at the time and Firestone found 40 acres of beanfield to house his new manufacturing plant . Architects Curlett and Beelman
created a spectacular four - storey Italianate complex , with its own power plant and gorgeous polychrome murals by Gladding McBean depicting the tire and
rubber - making process . A year after the plant opened in 1928 it doubled in size . By 1954 , when they added the Corporal guided missile to their offerings ,
the plant was nearly a million square feet . The town grew around Firestone , they named the main boulevard through town after Harvey , and Los Angeles became
the number one tire market in the country . By the mid-1970s Ford and GM had massive layoffs as Firestone and other manufacturers opened new plants in non-union
locales like Wilson , North Carolina . After much downsizing the end came in 1980 when 1,300 workers were laid off and the plant closed .
</item 9>

<item 10>
Firestone Tire and Rubber Company
Harvey Firestone had a friendship with Henry Ford . Firestone used this relationship to become the original equipment supplier of Ford Motor Company automobiles
, and was also active in the replacement market . In 1988 , the company was sold to the Japanese Bridgestone Corporation . Contents ( hide ) 1 History 1.1 Early
- to - mid-20th century 1.2 Restructuring and sale to Bridgestone 2 In motorsport 3 Corporate troubles 3.1 Great American streetcar scandal 3.2 Firestone 500
tread separation problem 3.3 Liberian rubber plantation 3.4 Ford Explorer rollover problem 4 Television advertisement jingle 5 Other brands 6 See also 7
References 8 External links History ( edit ) Early - to - mid-20th century ( edit ) The First Firestone store Firestone newspaper ad from 1919 , explaining the
benefits of their cord tire . Firestone was originally based in Akron , Ohio , also the hometown of its archrival , Goodyear Tire and Rubber Company , and
another two mid-sized competitors , General Tire and Rubber and BF Goodrich .
</item 10>
=== END ITEM id=8943781ef33850e8 ===

=== ITEM id=934c9d38edb0a31d ===
Gold sentences and facts:
Sentence 0: ESPN Monday Night Football ( abbreviated as MNF and also known as ESPN Monday Night Football on ABC for rare live special broadcast ) is a live television broadcast of weekly National Football League ( NFL ) games on ESPN in the United States . From 1970 to 2005 , it aired on sister broadcast network ABC . Monday Night Football was , along with Hallmark Hall of Fame and the Walt Disney anthology television series , one of the longest - running prime time programs ever on commercial network television , and one of the highest - rated , particularly among male viewers . MNF is preceded on air by Monday Night Countdown served by Applebee 's .
  Fact 0.0: ESPN Monday Night Football is abbreviated as MNF.
  Fact 0.1: ESPN Monday Night Football is also known as ESPN Monday Night Football on ABC for rare live special broadcast.
  Fact 0.2: ESPN Monday Night Football is a live television broadcast of weekly National Football League (NFL) games.
  Fact 0.3: ESPN Monday Night Football airs on ESPN in the United States.
  Fact 0.4: From 1970 to 2005, Monday Night Football aired on sister broadcast network ABC.
  Fact 0.5: Monday Night Football is one of the longest-running prime time programs ever on commercial network television.
  Fact 0.6: Monday Night Football is, along with Hallmark Hall of Fame and the Walt Disney anthology television series, one of the longest-running prime time programs ever on commercial network television.
  Fact 0.7: Monday Night Football is one of the highest-rated programs, particularly among male viewers.
  Fact 0.8: Monday Night Football is preceded on air by Monday Night Countdown.
  Fact 0.9: Monday Night Countdown is served by Applebee's.
Sentence 1: ESPN Monday Night Football Genre NFL game telecasts Created by Roone Arledge Directed by Chet Forte ( 1970 -- 1986 ) Larry Kamm ( 1987 ) Craig Janoff ( 1988 -- 1999 ) Drew Esocoff ( 2000 -- 2005 ) Chip Dean ( 2006 -- present ) Presented by Commentators : Sean McDonough Lisa Salters Studio hosts : Suzy Kolber Steve Young Randy Moss Matt Hasselbeck Charles Woodson Opening theme `` Score '' by Bob 's Band ( 1970 -- 1975 ) `` All My Rowdy Friends Are Here on Monday Night '' by Hank Williams , Jr. ( 1989 -- 2011 ; 2017 -- present ) `` Heavy Action '' by Johnny Pearson ( 1976 -- 1988 ; 2011 -- 2016 ) ( used as secondary theme from 1989 to 2011 and again from 2017 -- present ) Country of origin United States Original language ( s ) English No. of seasons 47 ( NFL seasons ) No. of episodes 718 ( games ) Production Executive producer ( s ) Norby Williamson ( 2006 -- present ) Producer ( s ) Roger Lewin Jay Rothman Suzy Kolber ( uncredited ; 2006 -- present ) Location ( s ) Various NFL stadiums ( game telecasts ) ESPN Center , Bristol , Connecticut ( studio segments ) Camera setup Multi-camera Running time 3 - 4 hours Production company ( s ) National Football League ABC Sports ( 1970 -- 2005 ) ESPN ( 2006 -- present ) Release Original network ABC ( 1970 -- 2005 ) ESPN ( 2006 -- present ) Picture format 480i ( SDTV ) ( 1970 -- 2010 ) , 480i ( 16 : 9 SDTV ) ( 2011 -- present ) , 720p ( HDTV ) ( 2011 -- present ) Original release September 21 , 1970 ( 1970 - 09 - 21 ) -- present Chronology Related shows Monday Night Countdown External links Website espn.go.com/nfl/mnf
  Fact 1.0: ESPN Monday Night Football has the genre NFL game telecasts.
  Fact 1.1: ESPN Monday Night Football was created by Roone Arledge.
  Fact 1.2: Chet Forte directed ESPN Monday Night Football from 1970 to 1986.
  Fact 1.3: Larry Kamm directed ESPN Monday Night Football in 1987.
  Fact 1.4: Craig Janoff directed ESPN Monday Night Football from 1988 to 1999.
  Fact 1.5: Drew Esocoff directed ESPN Monday Night Football from 2000 to 2005.
  Fact 1.6: Chip Dean has directed ESPN Monday Night Football from 2006 to the present.
  Fact 1.7: The commentators of ESPN Monday Night Football are Sean McDonough and Lisa Salters.
  Fact 1.8: The studio hosts of ESPN Monday Night Football are Suzy Kolber, Steve Young, Randy Moss, Matt Hasselbeck and Charles Woodson.
  Fact 1.9: The opening theme 'Score' by Bob's Band was used from 1970 to 1975.
  Fact 1.10: The opening theme 'All My Rowdy Friends Are Here on Monday Night' by Hank Williams, Jr. was used from 1989 to 2011 and again from 2017 to the present.
  Fact 1.11: The opening theme 'Heavy Action' by Johnny Pearson was used from 1976 to 1988 and from 2011 to 2016.
  Fact 1.12: 'Heavy Action' was used as the secondary theme from 1989 to 2011 and again from 2017 to the present.
  Fact 1.13: The country of origin of ESPN Monday Night Football is the United States.
  Fact 1.14: The original language of ESPN Monday Night Football is English.
  Fact 1.15: ESPN Monday Night Football has had 47 NFL seasons.
  Fact 1.16: ESPN Monday Night Football has had 718 games as episodes.
  Fact 1.17: Norby Williamson has been executive producer of ESPN Monday Night Football from 2006 to the present.
  Fact 1.18: Roger Lewin and Jay Rothman were producers of ESPN Monday Night Football.
  Fact 1.19: Suzy Kolber is an uncredited producer of ESPN Monday Night Football from 2006 to the present.
  Fact 1.20: ESPN Monday Night Football game telecasts are located at various NFL stadiums.
  Fact 1.21: ESPN Monday Night Football studio segments are produced at ESPN Center in Bristol, Connecticut.
  Fact 1.22: ESPN Monday Night Football uses a multi-camera setup.
  Fact 1.23: ESPN Monday Night Football has a running time of 3 to 4 hours.
  Fact 1.24: The National Football League is a production company of ESPN Monday Night Football.
  Fact 1.25: ABC Sports was a production company of ESPN Monday Night Football from 1970 to 2005.
  Fact 1.26: ESPN was a production company of ESPN Monday Night Football from 2006 to the present.
  Fact 1.27: The original network of ESPN Monday Night Football was ABC from 1970 to 2005.
  Fact 1.28: The original network of ESPN Monday Night Football is ESPN from 2006 to the present.
  Fact 1.29: ESPN Monday Night Football was broadcast in 480i (SDTV) picture format from 1970 to 2010.
  Fact 1.30: ESPN Monday Night Football was broadcast in 480i (16:9 SDTV) picture format from 2011 to the present.
  Fact 1.31: ESPN Monday Night Football was broadcast in 720p (HDTV) picture format from 2011 to the present.
  Fact 1.32: ESPN Monday Night Football originally released on September 21, 1970.
  Fact 1.33: ESPN Monday Night Football has been released from September 21, 1970 to the present.
  Fact 1.34: The related show of ESPN Monday Night Football is Monday Night Countdown.
  Fact 1.35: The website of ESPN Monday Night Football is espn.go.com/nfl/mnf.

Retrieved items:
<item 1>
Monday Night Football
wordpress \. com \ b on the global blacklist ESPN Monday Night Football Genre NFL game telecasts Created by Roone Arledge Directed by Chet Forte ( 1970 -- 1986
) Larry Kamm ( 1987 ) Craig Janoff ( 1988 -- 1999 ) Drew Esocoff ( 2000 -- 2005 ) Chip Dean ( 2006 -- present ) Presented by Commentators : Sean McDonough Lisa
Salters Studio hosts : Suzy Kolber Steve Young Randy Moss Matt Hasselbeck Charles Woodson Opening theme `` Score '' by Bob 's Band ( 1970 -- 1975 ) `` All My
Rowdy Friends Are Here on Monday Night '' by Hank Williams , Jr. ( 1989 -- 2011 ; 2017 -- present ) `` Heavy Action '' by Johnny Pearson ( 1976 -- 1988 ; 2011
-- 2016 ) ( used as secondary theme from 1989 to 2011 and again from 2017 -- present ) Country of origin United States Original language ( s ) English No. of
seasons 47 ( NFL seasons ) No.
</item 1>

<item 2>
Monday Night Football
1987 -- 2005 ) NFL Draft NFL on ABC Sports Football Sunday on ESPN Radio Studio programs The Blitz Monday Night Countdown NFL Matchup NFL Primetime Sunday NFL
Countdown NFL Live NFL Insiders Other football programs Arena Football League on ESPN CFL on TSN ESPN College Football USFL on ESPN Related articles History of
Monday Night Football NFL on television ( history ) Commentators Sean McDonough Play - by - play TBA Color commentary Lisa Salters Sideline reporter Former
commentators Bonnie Bernstein Mike Ditka Jon Gruden Ron Jaworski Suzy Kolber Tony Kornheiser Michele Tafoya Joe Theismann Mike Tirico Dick Vermeil Steve Young
Lore televised by ESPN ESPN takes over MNF 2005 Saints -- Giants / Cowboys -- Redskins Doubleheader 2006 Bears -- Cardinals Comeback Fail Mary Sunday Night
Football era American Bowl ( 1990 - 2005 ) Fútbol Americano Pro Bowls broadcast on ESPN 1988 1990 1991 1992 1994 2005 2006 2015 2016 2017 2018 Music Edd
Kalehoff Johnny Pearson Hank Williams , Jr . Songs `` All My Rowdy Friends Are Here on Monday Night '' `` Heavy
</item 2>

<item 3>
Monday Night Football
go.com/nfl/mnf ESPN Monday Night Football ( abbreviated as MNF and also known as ESPN Monday Night Football on ABC for rare live special broadcast ) is a live
television broadcast of weekly National Football League ( NFL ) games on ESPN in the United States . From 1970 to 2005 , it aired on sister broadcast network
ABC . Monday Night Football was , along with Hallmark Hall of Fame and the Walt Disney anthology television series , one of the longest - running prime time
programs ever on commercial network television , and one of the highest - rated , particularly among male viewers . MNF is preceded on air by Monday Night
Countdown served by Applebee 's . Monday Night Football is also broadcast in Canada on TSN and RDS , and in most of Europe . On September 7 , 2013 , the NFL
announced that British Eurosport would show Monday Night Football games live in the United Kingdom for the 2013 and 2014 seasons . The telecasts are also seen
in most of Australia on ESPN Australia , in Portugal on SportTV 3 and SportTV HD and on TV 2 Sport in Denmark , and in some other regions of the world outside
the U.S.
</item 3>

<item 4>
Monday Night Football
However , the forecasting abilities of the NFL 's scheduling officials have exhibited uncanny proficiency . Many MNF games occurring later in the season feature
at least one team that is either headed to the playoffs or must win the MNF game to clinch a playoff spot . The problem of having a national spotlight game
which during the season 's most critical weeks late in the year probably would not show the most important game of the week was long known by the league and
network . As a result of this , the NFL wished to move the `` Game of the Week '' idea to Sunday nights as to make flex scheduling possible . This was a move
which would ultimately mean the end of Monday Night Football on ABC ( cable games are protected from the NFL 's flexible scheduling rule adopted for the 2006 --
07 season ; the new rule applies only to CBS , Fox , and NBC 's Sunday night games ) . The league currently has no interest in expanding flexible scheduling to
include Monday Night Football , citing the logistical issues of moving games back and forth between Sundays and Mondays . Franchises with the most appearances (
edit ) The franchise with the most Monday night appearances is the Miami Dolphins .
</item 4>

<item 5>
Monday Night Football
From 2003 to 2005 , one game was played on Thursday and another Monday under the Monday Night Football banner . Starting in 2006 , when the series moved to
cable , two games are played on the opening Monday night to capitalize on fan interest during `` NFL Kickoff Weekend '' . Monday night games early in the season
are often highly anticipated since records are new , teams usually are showcasing fresh talent and potential , and storylines coming into the season are often
played out as fans try to see if these hyped teams are up to form . Since no one knows during the first month of the season if a team is indeed good , or will
rebound from a difficult start , interest is usually high for the first few weeks of the MNF season . Since the MNF schedule is set in April and can not be
changed , the league and network can not guarantee a late season matchup will have any significance or be highly anticipated .
</item 5>

<item 6>
Monday Night Football
The layout is standard , with teams on the background of their color , with timeout indicators also displaying in team colors . Experimentally , the score bar
`` auto - hides '' itself during plays , though it is shown permanently during the final two minutes of each half and , if necessary , overtime . This is also
the first time that ESPN has used the full 16 : 9 frame for its NFL graphics . Scoring records ( edit ) See also : List of Monday Night Football results ( 1970
-- 89 ) , List of Monday Night Football results ( 1990 -- 2009 ) , and List of Monday Night Football results ( 2010 -- present ) Most points 59 -- Philadelphia
Eagles , November 15 , 2010 vs. Washington Redskins 55 -- Indianapolis Colts , October 31 , 1988 vs. Denver Broncos 52 -- San Francisco 49ers , December 23 ,
1991 vs. Chicago Bears 52 -- San Diego Chargers , December 20 , 1982 vs. Cincinnati Bengals 51 -- New Orleans Saints , November 24 , 2008 vs.
</item 6>

<item 7>
Monday Night Football
6 2016 summary 2.2. 7 2017 summary 3 Theme music 4 Digital on - screen graphics 4.1 1997 -- 2005 4.2 2006 -- present 5 Scoring records 6 Scheduling on local
stations 7 Commentators 8 Foreign - language versions 8.1 Spanish version 8.2 Portuguese version 9 Radio broadcasts 10 Nielsen ratings 11 Additional NFL game
rights 11.1 Playoff games 11.2 Pro Bowl 12 See also 13 Footnotes 14 References 15 Further reading 16 External links Overview ( edit ) This section needs
expansion . You can help by adding to it . ( November 2010 ) See also : History of Monday Night Football After 45 seasons , there now have been a total of 730
games televised by the Monday Night Football franchise . Scheduling problems ( edit ) To avoid any scheduling unfairness where , just before the first playoff
game , a team may have five days off and others six , there is no Monday night game during the final week of the regular season .
</item 7>

<item 8>
Monday Night Football
Jr . Songs `` All My Rowdy Friends Are Here on Monday Night '' `` Heavy Action '' `` Start Me Up '' `` The Party 's Over '' Yearly results Monday night NFL
games prior to 1970 1970 -- 1989 1990 -- 2009 2010 -- present National Football League on television and radio Broadcast partners ABC ( AFL ) CBS Fox NBC
Defunct networks DuMont Sports Network Monday Night Football History Monday Night Countdown Results Games prior to 1970 1970 -- 1989 1990 -- 2009 2010 --
present Sunday Night Football ESPN NBC TNT Results ESPN NBC TNT Pregame TV programs The NFL Today CBS Fox NFL Sunday ESPN Sunday NFL Countdown Monday Night
Countdown NBC The NFL on NBC pregame show Football Night in America NFL Network List of programs NFL AM NFL Classics NFL GameDay NFL Replay NFL Total Access The
Timeline Thursday Night Football Results ( 2006 -- present ) 2007 New England Patriots -- New York Giants game Miracle in Motown NFL Films TV programs Football
Follies A Football Life Full Color Football Hard Knocks Inside the NFL NFL Dream Season NFL Films Game of the Week NFL 's Greatest Games NFL Matchup NFL Top 10
NFL Top 100 Other
</item 8>

<item 9>
*Monday Night Football* is ESPN’s long-running NFL game telecast, created by Roone Arledge and presented in English in the United States. The excerpt lists 47
NFL seasons and 730 games after 45 seasons. Current commentators are Sean McDonough and Lisa Salters; studio hosts include Suzy Kolber and former players. Its
theme music has included “Heavy Action” and Hank Williams Jr.’s “All My Rowdy Friends Are Here on Monday Night.” No game is scheduled on
</item 9>

<item 10>
Monday Night Football
As of 2008 , Westwood One was no longer controlled by CBS , but the network retained its NFL broadcast rights . In 2011 , Westwood One was purchased by Dial
Global . Then in 2014 , Dial Global , including Westwood One , was acquired by Cumulus Media . The Spanish - language broadcast is carried on ESPN Deportes
Radio . Nielsen ratings ( edit ) This article needs to be updated . Please update this article to reflect recent events or newly available information . (
December 2016 ) The highest - rated Monday Night Football telecast on ABC was the Miami Dolphins ' victory over the previously undefeated Chicago Bears on
December 2 , 1985 , which drew a national Nielsen rating of 29.6 and a share of 46 . ABC 's lowest - rated MNF game was the St. Louis Rams ' defeat of the Tampa
Bay Buccaneers on October 18 , 2004 , which drew a 7.7 rating . During the 1995 -- 1996 television season , Monday Night Football averaged a 17.1 household
rating . Its main competitor , Murphy Brown on CBS , averaged a 12.3 rating .
</item 10>
=== END ITEM id=934c9d38edb0a31d ===

=== ITEM id=fd781d0468ca65b6 ===
Gold sentences and facts:
Sentence 0: Within eukaryotic cells , DNA is organized into long structures called chromosomes . During cell division these chromosomes are duplicated in the process of DNA replication , providing each cell its own complete set of chromosomes . Eukaryotic organisms ( animals , plants , fungi and protists ) store most of their DNA inside the cell nucleus and some of their DNA in organelles , such as mitochondria or chloroplasts . In contrast prokaryotes ( bacteria and archaea ) store their DNA only in the cytoplasm . Within the eukaryotic chromosomes , chromatin proteins such as histones compact and organize DNA . These compact structures guide the interactions between DNA and other proteins , helping control which parts of the DNA are transcribed .
  Fact 0.0: Within eukaryotic cells, DNA is organized into long structures called chromosomes.
  Fact 0.1: During cell division, chromosomes are duplicated in the process of DNA replication.
  Fact 0.2: DNA replication during cell division provides each cell its own complete set of chromosomes.
  Fact 0.3: Eukaryotic organisms (animals, plants, fungi and protists) store most of their DNA inside the cell nucleus.
  Fact 0.4: Eukaryotic organisms store some of their DNA in organelles, such as mitochondria or chloroplasts.
  Fact 0.5: Prokaryotes (bacteria and archaea) store their DNA only in the cytoplasm, in contrast to eukaryotes.
  Fact 0.6: Within eukaryotic chromosomes, chromatin proteins such as histones compact and organize DNA.
  Fact 0.7: The compact structures formed by chromatin proteins guide the interactions between DNA and other proteins.
  Fact 0.8: The compact structures formed by chromatin proteins help control which parts of the DNA are transcribed.
Sentence 1: Genomic DNA is tightly and orderly packed in the process called DNA condensation , to fit the small available volumes of the cell . In eukaryotes , DNA is located in the cell nucleus , with small amounts in mitochondria and chloroplasts . In prokaryotes , the DNA is held within an irregularly shaped body in the cytoplasm called the nucleoid . The genetic information in a genome is held within genes , and the complete set of this information in an organism is called its genotype . A gene is a unit of heredity and is a region of DNA that influences a particular characteristic in an organism . Genes contain an open reading frame that can be transcribed , and regulatory sequences such as promoters and enhancers , which control transcription of the open reading frame .
  Fact 1.0: Genomic DNA is tightly and orderly packed in a process called DNA condensation.
  Fact 1.1: DNA condensation fits genomic DNA into the small available volumes of the cell.
  Fact 1.2: In eukaryotes, DNA is located in the cell nucleus, with small amounts in mitochondria and chloroplasts.
  Fact 1.3: In prokaryotes, DNA is held within an irregularly shaped body in the cytoplasm called the nucleoid.
  Fact 1.4: The genetic information in a genome is held within genes.
  Fact 1.5: The complete set of genetic information in an organism is called its genotype.
  Fact 1.6: A gene is a unit of heredity.
  Fact 1.7: A gene is a region of DNA that influences a particular characteristic in an organism.
  Fact 1.8: Genes contain an open reading frame that can be transcribed.
  Fact 1.9: Genes contain regulatory sequences such as promoters and enhancers.
  Fact 1.10: The regulatory sequences such as promoters and enhancers control transcription of the open reading frame.
Sentence 2: DNA usually occurs as linear chromosomes in eukaryotes , and circular chromosomes in prokaryotes . The set of chromosomes in a cell makes up its genome ; the human genome has approximately 3 billion base pairs of DNA arranged into 46 chromosomes . The information carried by DNA is held in the sequence of pieces of DNA called genes . Transmission of genetic information in genes is achieved via complementary base pairing . For example , in transcription , when a cell uses the information in a gene , the DNA sequence is copied into a complementary RNA sequence through the attraction between the DNA and the correct RNA nucleotides . Usually , this RNA copy is then used to make a matching protein sequence in a process called translation , which depends on the same interaction between RNA nucleotides . In alternative fashion , a cell may simply copy its genetic information in a process called DNA replication . The details of these functions are covered in other articles ; here the focus is on the interactions between DNA and other molecules that mediate the function of the genome .
  Fact 2.0: DNA usually occurs as linear chromosomes in eukaryotes.
  Fact 2.1: DNA usually occurs as circular chromosomes in prokaryotes.
  Fact 2.2: The set of chromosomes in a cell makes up its genome.
  Fact 2.3: The human genome has approximately 3 billion base pairs of DNA.
  Fact 2.4: The human genome's DNA is arranged into 46 chromosomes.
  Fact 2.5: The information carried by DNA is held in the sequence of pieces of DNA called genes.
  Fact 2.6: Transmission of genetic information in genes is achieved via complementary base pairing.
  Fact 2.7: In transcription, when a cell uses the information in a gene, the DNA sequence is copied into a complementary RNA sequence.
  Fact 2.8: The copying in transcription happens through attraction between the DNA and the correct RNA nucleotides.
  Fact 2.9: The RNA copy is usually then used to make a matching protein sequence in a process called translation.
  Fact 2.10: Translation depends on the same interaction between RNA nucleotides.
  Fact 2.11: In an alternative fashion, a cell may simply copy its genetic information in a process called DNA replication.
  Fact 2.12: The details of these functions are covered in other articles.
  Fact 2.13: The focus of this text is on the interactions between DNA and other molecules that mediate the function of the genome.

Retrieved items:
<item 1>
DNA
This information is replicated as and when the two strands separate . A large part of DNA ( more than 98 % for humans ) is non-coding , meaning that these
sections do not serve as patterns for protein sequences . The two strands of DNA run in opposite directions to each other and are thus antiparallel . Attached
to each sugar is one of four types of nucleobases ( informally , bases ) . It is the sequence of these four nucleobases along the backbone that encodes
biological information . RNA strands are created using DNA strands as a template in a process called transcription . Under the genetic code , these RNA strands
are translated to specify the sequence of amino acids within proteins in a process called translation . Within eukaryotic cells , DNA is organized into long
structures called chromosomes . During cell division these chromosomes are duplicated in the process of DNA replication , providing each cell its own complete
set of chromosomes . Eukaryotic organisms ( animals , plants , fungi and protists ) store most of their DNA inside the cell nucleus and some of their DNA in
organelles , such as mitochondria or chloroplasts .
</item 1>

<item 2>
DNA
The DNA sequence may be aligned with other DNA sequences to identify homologous sequences and locate the specific mutations that make them distinct . These
techniques , especially multiple sequence alignment , are used in studying phylogenetic relationships and protein function . Data sets representing entire
genomes ' worth of DNA sequences , such as those produced by the Human Genome Project , are difficult to use without the annotations that identify the locations
of genes and regulatory elements on each chromosome . Regions of DNA sequence that have the characteristic patterns associated with protein - or RNA - coding
genes can be identified by gene finding algorithms , which allow researchers to predict the presence of particular gene products and their possible functions in
an organism even before they have been isolated experimentally . Entire genomes may also be compared , which can shed light on the evolutionary history of
particular organism and permit the examination of complex evolutionary events . DNA nanotechnology The DNA structure at left ( schematic shown ) will self -
assemble into the structure visualized by atomic force microscopy at right . DNA nanotechnology is the field that seeks to design nanoscale structures using the
molecular recognition properties of DNA molecules . Image from Strong , 2004 .
</item 2>

<item 3>
DNA
`` Jump up ^ Miescher F ( 1871 ) . `` Ueber die chemische Zusammensetzung der Eiterzellen '' ( On the chemical composition of pus cells ) . Medicinisch -
chemische Untersuchungen ( in German ) . 4 : 441 -- 460 . Ich habe mich daher später mit meinen Versuchen an die ganzen Kerne gehalten , die Trennung der Körper
, die ich einstweilen ohne weiteres Präjudiz als lösliches und unlösliches Nuclein bezeichnen will , einem günstigeren Material überlassend . ( Therefore , in
my experiments I subsequently limited myself to the whole nucleus , leaving to a more favorable material the separation of the substances , that for the present
, without further prejudice , I will designate as soluble and insoluble nuclear material ( `` Nuclein '' ) Jump up ^ Dahm R ( January 2008 ) . `` Discovering
DNA : Friedrich Miescher and the early years of nucleic acid research '' . Human Genetics . 122 ( 6 ) : 565 -- 81 .
</item 3>

<item 4>
DNA
Genes and genomes Further information : Cell nucleus , Chromatin , Chromosome , Gene , and Noncoding DNA Genomic DNA is tightly and orderly packed in the
process called DNA condensation , to fit the small available volumes of the cell . In eukaryotes , DNA is located in the cell nucleus , with small amounts in
mitochondria and chloroplasts . In prokaryotes , the DNA is held within an irregularly shaped body in the cytoplasm called the nucleoid . The genetic
information in a genome is held within genes , and the complete set of this information in an organism is called its genotype . A gene is a unit of heredity and
is a region of DNA that influences a particular characteristic in an organism . Genes contain an open reading frame that can be transcribed , and regulatory
sequences such as promoters and enhancers , which control transcription of the open reading frame . In many species , only a small fraction of the total
sequence of the genome encodes protein . For example , only about 1.5 % of the human genome consists of protein - coding exons , with over 50 % of human DNA
consisting of non-coding repetitive sequences .
</item 4>

<item 5>
DNA
In this way , the base on the old strand dictates which base appears on the new strand , and the cell ends up with a perfect copy of its DNA . Extracellular
nucleic acids Naked extracellular DNA ( eDNA ) , most of it released by cell death , is nearly ubiquitous in the environment . Its concentration in soil may be
as high as 2 μg / L , and its concentration in natural aquatic environments may be as high at 88 μg / L. Various possible functions have been proposed for eDNA
: it may be involved in horizontal gene transfer ; it may provide nutrients ; and it may act as a buffer to recruit or titrate ions or antibiotics .
Extracellular DNA acts as a functional extracellular matrix component in the biofilms of several bacterial species . It may act as a recognition factor to
regulate the attachment and dispersal of specific cell types in the biofilm ; it may contribute to biofilm formation ; and it may contribute to the biofilm 's
physical strength and resistance to biological stress . Cell - free fetal DNA is found in the blood of the mother , and can be sequenced to determine a great
deal of information about the developing fetus .
</item 5>

<item 6>
DNA
This can be used in studies ranging from ecological genetics to anthropology ; For example , DNA evidence is being used to try to identify the Ten Lost Tribes
of Israel . Information storage Main article : DNA digital data storage In a paper published in Nature in January 2013 , scientists from the European
Bioinformatics Institute and Agilent Technologies proposed a mechanism to use DNA 's ability to code information as a means of digital data storage . The group
was able to encode 739 kilobytes of data into DNA code , synthesize the actual DNA , then sequence the DNA and decode the information back to its original form
, with a reported 100 % accuracy . The encoded information consisted of text files and audio files . A prior experiment was published in August 2012 . It was
conducted by researchers at Harvard University , where the text of a 54,000 - word book was encoded in DNA . Moreover , in living cells , the storage can be
turned active by enzymes . Light - gated protein domains fused to DNA processing enzymes are suitable for that task in vitro . Fluorescent exonucleases can
transmit the output according to the nucleotide they have read .
</item 6>

<item 7>
DNA
Biological functions Location of eukaryote nuclear DNA within the chromosomes . DNA usually occurs as linear chromosomes in eukaryotes , and circular
chromosomes in prokaryotes . The set of chromosomes in a cell makes up its genome ; the human genome has approximately 3 billion base pairs of DNA arranged into
46 chromosomes . The information carried by DNA is held in the sequence of pieces of DNA called genes . Transmission of genetic information in genes is achieved
via complementary base pairing . For example , in transcription , when a cell uses the information in a gene , the DNA sequence is copied into a complementary
RNA sequence through the attraction between the DNA and the correct RNA nucleotides . Usually , this RNA copy is then used to make a matching protein sequence
in a process called translation , which depends on the same interaction between RNA nucleotides . In alternative fashion , a cell may simply copy its genetic
information in a process called DNA replication . The details of these functions are covered in other articles ; here the focus is on the interactions between
DNA and other molecules that mediate the function of the genome .
</item 7>

<item 8>
DNA
In contrast prokaryotes ( bacteria and archaea ) store their DNA only in the cytoplasm . Within the eukaryotic chromosomes , chromatin proteins such as histones
compact and organize DNA . These compact structures guide the interactions between DNA and other proteins , helping control which parts of the DNA are
transcribed . DNA was first isolated by Friedrich Miescher in 1869 . Its molecular structure was first identified by James Watson and Francis Crick at the
Cavendish Laboratory within the University of Cambridge in 1953 , whose model - building efforts were guided by X-ray diffraction data acquired by Raymond
Gosling , who was a post-graduate student of Rosalind Franklin . DNA is used by researchers as a molecular tool to explore physical laws and theories , such as
the ergodic theorem and the theory of elasticity . The unique material properties of DNA have made it an attractive molecule for material scientists and
engineers interested in micro - and nano - fabrication . Among notable advances in this field are DNA origami and DNA - based hybrid materials .
</item 8>

<item 9>
DNA
This does not appear to be the entire story . Four modifications to the cytosine residues in human DNA have been reported . These modifications are the addition
of methyl ( CH ) - , hydroxymethyl ( CH OH ) - , formyl ( CHO ) - and carboxyl ( COOH ) - groups . These modifications are thought to have regulatory functions
. Listing of non canonical bases found in DNA Seventeen non canonical bases are known to occur in DNA . Most of these are modifications of the canonical bases
plus uracil .
</item 9>

<item 10>
DNA
Claims for older DNA have been made , most notably a report of the isolation of a viable bacterium from a salt crystal 250 million years old , but these claims
are controversial . Building blocks of DNA ( adenine , guanine , and related organic molecules ) may have been formed extraterrestrially in outer space .
Complex DNA and RNA organic compounds of life , including uracil , cytosine , and thymine , have also been formed in the laboratory under conditions mimicking
those found in outer space , using starting chemicals , such as pyrimidine , found in meteorites . Pyrimidine , like polycyclic aromatic hydrocarbons ( PAHs ) ,
the most carbon - rich chemical found in the universe , may have been formed in red giants or in interstellar cosmic dust and gas clouds . Uses in technology
Genetic engineering Further information : Molecular biology , Nucleic acid methods , and Genetic engineering Methods have been developed to purify DNA from
organisms , such as phenol - chloroform extraction , and to manipulate it in the laboratory , such as restriction digests and the polymerase chain reaction .
Modern biology and biochemistry make intensive use of these techniques in recombinant DNA technology .
</item 10>
=== END ITEM id=fd781d0468ca65b6 ===
