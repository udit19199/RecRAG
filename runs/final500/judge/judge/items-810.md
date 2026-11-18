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

=== ITEM id=e3df4ca9f5f9a676 ===
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
</item 1>

<item 2>
Adobe Flash
, blend modes ( similar to Adobe Photoshop ) , and advanced features for FLV video . Adobe ( edit ) Macromedia was acquired by Adobe Systems in 2005 , and the
entire Macromedia product line including Flash , Dreamweaver , Director / Shockwave , Fireworks ( which has since been discontinued ) and Authorware is now
handled by Adobe . In 2007 , Adobe 's first version release was Adobe Flash CS3 Professional , the ninth major version of Flash . It introduced the ActionScript
3.0 programming language , which supported modern programming practices and enabled business applications to be developed with Flash . Adobe Flex Builder (
built on Eclipse ) targeted the enterprise application development market , and was also released the same year . Flex Builder included the Flex SDK , a set of
components that included charting , advanced UI , and data services ( Flex Data Services ) . In 2008 , Adobe released the tenth version of Flash , Adobe Flash
CS4 . Flash 10 improved animation capabilities within the Flash editor , adding a motion editor panel ( similar to Adobe After Effects ) , inverse kinematics (
bones ) , basic 3D object animation , object - based animation , and other text and graphics features .
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
The Apache Flex ActionScript 3.0 compiler generates SWF files from ActionScript 3 files . Flex was the primary ActionScript 3 compiler and was actively
developed by Adobe before it was donated to Apache Software Foundation in 2011 . Adobe Animate -- primarily used to design graphics and animation , but supports
ActionScript scripting and debugging . Adobe Flash Builder -- enterprise application development & debugging , contains the Flex SDK with UI and charting
components . Adobe Scout -- a visual profiler to optimize performance of Flash content . CrossBridge -- a free SDK to cross-compile C++ code to run in Flash
Player . Third - party tools ( edit ) Third - party development tools have been created to assist developers in creating software applications and video games
with Flash . FlashDevelop is a free and open - source Flash ActionScript IDE , which includes a project manager and debugger for building applications on Flash
Player and Adobe AIR . Powerflasher FDT is a commercial ActionScript IDE similar to FlashDevelop . Haxe is an open source , high - level object - oriented
programming language geared towards web - content creation that can compile SWF files from Haxe programs .
</item 4>

<item 5>
Adobe Flash
Previously , developers could not use the specification for making SWF - compatible players , but only for making SWF - exporting authoring software . The
specification still omits information on codecs such as Sorenson Spark , however . Animation tools ( edit ) Official tools ( edit ) Main article : Adobe Animate
The Adobe Animate authoring program is primarily used to design graphics and animation and publish the same for websites , web applications , and video games .
The program also offers limited support for audio and video embedding , and ActionScript scripting . Adobe released Adobe LiveMotion , designed to create
interactive animation content and export it to a variety of formats , including SWF . LiveMotion failed to gain any notable user base . In February 2003 ,
Macromedia purchased Presedia , which had developed a Flash authoring tool that automatically converted PowerPoint files into Flash . Macromedia subsequently
released the new product as Breeze , which included many new enhancements .
</item 5>

<item 6>
Adobe Flash
Adobe Flash - Wikipedia Adobe Flash Jump to : navigation , search For the application / animation - maker , see Adobe Animate . For other uses , see Adobe Flash
( disambiguation ) . Adobe Flash Developer ( s ) Adobe Systems Macromedia FutureWave Target platform ( s ) Web browsers , iOS , Android , Windows , macOS Editor
software Adobe Animate , Flash Builder , FlashDevelop , Powerflasher FDT , Flash Catalyst Scaleform Player software Flash Player Adobe AIR OpenFL Scaleform
Gnash Lightspark Format ( s ) SWF FLV FLA Programming language ( s ) ActionScript Application ( s ) Animation Browser games Rich Internet applications Desktop
applications Mobile apps Mobile games console games computer games Status Active ( Until 2020 ) License Proprietary Adobe Flash is a multimedia software
platform used for production of animations , rich Internet applications , desktop applications , mobile applications , mobile games and embedded web browser
video players . Flash displays text , vector graphics and raster graphics to provide animations , video games and applications . It allows streaming of audio
and video , and can capture mouse , keyboard , microphone and camera input . Artists may produce Flash graphics and animations using Adobe Animate .
</item 6>

<item 7>
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
</item 7>

<item 8>
Adobe Flash Player
( SWC ) Software ( list ) Players Adobe AIR Adobe Flash Player OpenFL Gnash Lightspark Gameswf Tamarin Animation tools Adobe Animate Toon Boom Moho GoAnimate
Stencyl Programming tools FlashDevelop Powerflasher FDT Apache Flex Haxe CrossBridge Adobe Scout FlashFirebug Libraries Starling Framework Away3D Flare3D
Stage3D PureMVC Ming List of Flex frameworks Converters Adobe Flash Media Live Encoder SWFTools swfmill Google Swiffy Adobe Wallaby Server - side Adobe Flash
Media Server Obsolete software Adobe Flash Builder Adobe Flash Catalyst Adobe Flash Lite SWFAddress SWFObject MTASC Scaleform GFx SWiSH Max Swift 3D
Papervision3D Ajax Animator Shumway Swfdec OpenLaszlo Related topics Comparison of HTML5 and Flash Apple and Adobe Flash controversy Flash animation films
television Protected Streaming Real - Time Messaging Protocol Real - Time Media Flow Protocol Developers FutureWave Software Macromedia Adobe Systems Adobe
Systems Software Suites Creative Cloud ( Creative Suite ) eLearning Suite Technical Communication Suite Desktop Animate Acrobat After Effects Brackets Edge Code
Edge Reflow Digital Editions Director Dreamweaver FrameMaker FreeHand Flash Player Flex Builder GoLive Illustrator InDesign Media Encoder Media Player Muse
</item 8>

<item 9>
Adobe Flash
Artists may produce Flash graphics and animations using Adobe Animate . Software developers may produce applications and video games using Adobe Flash Builder ,
FlashDevelop , Flash Catalyst , or any text editor when used with the Apache Flex SDK . End - users can view Flash content via Flash Player ( for web browsers )
, AIR ( for desktop or mobile apps ) or third - party players such as Scaleform ( for video games ) . Adobe Flash Player ( supported on Microsoft Windows ,
macOS and Linux ) enables end - users to view Flash content using web browsers . Adobe Flash Lite enabled viewing Flash content on older smartphones , but has
been discontinued and superseded by Adobe AIR . The ActionScript programming language allows the development of interactive animations , video games , web
applications , desktop applications and mobile applications . Programmers can implement Flash software using an IDE such as Adobe Animate , Adobe Flash Builder
, Adobe Director , FlashDevelop and Powerflasher FDT . Adobe AIR enables full - featured desktop and mobile applications to be developed with Flash and
published for Windows , macOS , Android , iOS , Xbox One , PlayStation 4 , and Nintendo Wii U and Switch .
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
=== END ITEM id=e3df4ca9f5f9a676 ===

=== ITEM id=b7d739078ddebcba ===
Gold sentences and facts:
Sentence 0: His roles continued in the 1990s with the TV series Cracker ( 1993 -- 1996 , returning in 2006 for a one - off special ) , in which he starred as forensic psychologist Dr. Edward `` Fitz '' Fitzgerald . The role won him three BAFTA awards . Roles in bigger films followed ; the James Bond films GoldenEye ( 1995 ) and The World Is Not Enough ( 1999 ) , a supporting role in From Hell ( 2001 ) , as well as half - giant Rubeus Hagrid in the Harry Potter films ( 2001 -- 2011 ) . J.K. Rowling , author of the Harry Potter series , had Coltrane at the top of her list to play Hagrid and , when asked whom she would like to see in the role , responded `` Robbie Coltrane for Hagrid '' in one quick breath . There is also an unnamed 8ft actor who stands in for the 6ft 1in Coltrane in some scenes .
  Fact 0.0: His roles continued in the 1990s.
  Fact 0.1: He appeared in the TV series Cracker from 1993 to 1996.
  Fact 0.2: He returned to the TV series Cracker in 2006 for a one-off special.
  Fact 0.3: In Cracker he starred as forensic psychologist Dr. Edward "Fitz" Fitzgerald.
  Fact 0.4: The role in Cracker won him three BAFTA awards.
  Fact 0.5: Roles in bigger films followed.
  Fact 0.6: He appeared in the James Bond film GoldenEye (1995).
  Fact 0.7: He appeared in the James Bond film The World Is Not Enough (1999).
  Fact 0.8: He had a supporting role in From Hell (2001).
  Fact 0.9: He played half-giant Rubeus Hagrid in the Harry Potter films from 2001 to 2011.
  Fact 0.10: J.K. Rowling, author of the Harry Potter series, had him at the top of her list to play Hagrid.
  Fact 0.11: When asked whom she would like to see in the role, Rowling responded "Robbie Coltrane for Hagrid" in one quick breath.
  Fact 0.12: An unnamed 8ft actor stands in for the 6ft 1in Coltrane in some scenes.
Sentence 1: Robbie Coltrane OBE Coltrane in April 2007 Anthony Robert McMillan ( 1950 - 03 - 30 ) 30 March 1950 ( age 67 ) Rutherglen , South Lanarkshire , Scotland Nationality British Alma mater University of Edinburgh Occupation Actor , Author Years active 1979 -- present Height 6 ft 1 in ( 185 cm ) Spouse ( s ) Rhona Gemmell ( m . 1999 -- 2003 ) Children
  Fact 1.0: Robbie Coltrane holds the OBE honour.
  Fact 1.1: A photo caption shows Coltrane in April 2007.
  Fact 1.2: Coltrane's birth name is Anthony Robert McMillan.
  Fact 1.3: Coltrane was born on 30 March 1950.
  Fact 1.4: Coltrane is listed as age 67.
  Fact 1.5: Coltrane was born in Rutherglen, South Lanarkshire, Scotland.
  Fact 1.6: Coltrane's nationality is British.
  Fact 1.7: Coltrane's alma mater is the University of Edinburgh.
  Fact 1.8: Coltrane's occupations are actor and author.
  Fact 1.9: Coltrane has been active as a performer from 1979 to the present.
  Fact 1.10: Coltrane's height is 6 ft 1 in (185 cm).
  Fact 1.11: Coltrane's spouse was Rhona Gemmell, married from 1999 to 2003.
  Fact 1.12: The children field is listed with no value.

Retrieved items:
<item 1>
Robbie Coltrane
Bill Lintz 1998 Frogs for Snakes Al 1998 Montana The Boss 1999 World Is Not Enough , The The World Is Not Enough Valentin Dmitrovich Zukovsky 1999 Message in a
Bottle Charlie Toschi 2001 Harry Potter and the Philosopher 's Stone Rubeus Hagrid Nominated -- BAFTA Award for Best Actor in a Supporting Role Nominated --
Saturn Award for Best Supporting Actor 2001 On the Nose Delaney 2001 From Hell Sergeant Peter Godley 2002 Harry Potter and the Chamber of Secrets Rubeus Hagrid
Nominated -- Phoenix Film Critics Society Award for Best Ensemble Acting Ocean 's Twelve Matsui Harry Potter and the Prisoner of Azkaban Rubeus Hagrid Van
Helsing : The London Assignment Mr. Hyde voice Van Helsing Mr.
</item 1>

<item 2>
Robbie Coltrane
) Bill Nighy ( 2004 ) Rhys Ifans ( 2005 ) Mark Rylance ( 2006 ) Jim Broadbent ( 2007 ) Andrew Garfield ( 2008 ) Stephen Dillane ( 2009 ) Kenneth Branagh ( 2010
) Daniel Rigby ( 2011 ) Dominic West ( 2012 ) Ben Whishaw ( 2013 ) Sean Harris ( 2014 ) Jason Watkins ( 2015 ) Mark Rylance ( 2016 ) Adeel Akhtar ( 2017 ) VIAF
: 107686124 LCCN : no95038681 ISNI : 0000 0001 1002 5330 GND : 140684794 SUDOC : 110194942 BNF : cb14031105k ( data ) SNAC : w6320rb1 Retrieved from ``
https://en.wikipedia.org/w/index.php?title=Robbie_Coltrane&oldid=822199192 '' Categories : 21st - century Scottish male actors 1950 births Living people Alumni
of the University of Edinburgh BAFTA winners ( people ) Alumni of the Glasgow School of Art The
</item 2>

<item 3>
Robbie Coltrane
Hyde voice Van Helsing Mr. Hyde voice 2005 Harry Potter and the Goblet of Fire Rubeus Hagrid 2006 Stormbreaker The Prime Minister of Europe 2006 Provoked Lord
Edward Foster 2007 Harry Potter and the Order of the Phoenix Rubeus Hagrid 2008 Tale of Despereaux , The The Tale of Despereaux Gregory voice 2008 Brothers
Bloom , The The Brothers Bloom The Curator 2009 Harry Potter and the Half - Blood Prince Rubeus Hagrid 2009 Gooby Gooby voice Harry Potter and the Deathly
Hallows -- Part 1 Rubeus Hagrid 2011 Harry Potter and the Deathly Hallows -- Part 2 Rubeus Hagrid 2012 Brave Lord Dingwall voice Great Expectations Mr.
</item 3>

<item 4>
Robbie Coltrane
Retrieved 15 August 2011 . ( Accio Quote ! , the Largest Archive of J.K. Rowling quotes on the web ) Jump up ^ Ezard , John ( 5 June 2002 ) . `` Size means
everything as Harry 's big secret slips out '' . The Guardian . Retrieved 1 October 2016 . Jump up ^ Femia , Will ( 13 November 2001 ) . `` Chat Flashback : The
cinematic debut of Rubeus Hagrid '' . MSNBC . NBC News . Retrieved 1 October 2016 . In real life , I 'm 6'1 '' -- each way pretty well , north and south and
east and west , unfortunately Jump up ^ The Robbie Coltrane Special at the Internet Movie Database Jump up ^ `` The Gruffalo 's Child '' . BBC One . Retrieved
10 December 2011 . External links ( edit ) Wikimedia Commons has media related to Robbie Coltrane .
</item 4>

<item 5>
Robbie Coltrane
External links ( edit ) Wikimedia Commons has media related to Robbie Coltrane . Robbie Coltrane on IMDb Robbie Coltrane biography and credits at the British
Film Institute 's Screenonline The Comic Strip Ade Edmondson Rik Mayall Nigel Planer Peter Richardson Dawn French Jennifer Saunders Alexei Sayle Keith Allen
Robbie Coltrane Pete Richens Select episodes and films Five Go Mad in Dorset The Bullshitters : Roll Out The Gunbarrel The Supergrass Eat the Rich The Strike
Mr. Jolly Lives Next Door The Yob Did n't You Kill My Brother ? South Atlantic Raiders GLC : The Carnage Continues ...
</item 5>

<item 6>
Robbie Coltrane
Douglas Hain 2009 The Gruffalo The Gruffalo TV film ( children 's ) , voice 2011 Lead Balloon Donald Series 4 Episode 4 : `` Off '' Series 4 Episode 5 : ``
Blade '' Series 4 Episode 6 : `` End '' 50 Greatest Harry Potter Moments Himself Narrator The Gruffalo 's Child The Gruffalo TV film ( children 's ) , voice
2016 National Treasure Paul Finchley TV film Nominated -- British Academy Television Award for Best Actor Other Awards ( edit ) Won Evening Standard British
Film Award -- Peter Sellers Award for Comedy 1990 . He was awarded the OBE ( Officer of the Order of the British Empire ) in the 2006 New Year Honours for his
services to drama . In 2011 was honoured for his `` outstanding contribution '' to film as the British Academy Scotland Awards ( `` BAFTA Scotland Awards '' ) .
References ( edit ) Jump up ^ `` Robbie Coltrane Biography ( 1950 -- ) '' . Filmreference.com . Retrieved 15 August 2011 . Jump up ^ `` Robbie Coltrane 's
magical career '' . BBC News .
</item 6>

<item 7>
Robbie Coltrane
Contents ( hide ) 1 Early life and education 2 Career 3 Personal life 4 Filmography and awards 4.1 Film 4.2 Television 4.3 Other awards 5 References 6 External
links Early life and Education ( edit ) Coltrane was born Anthony Robert McMillan in Rutherglen , South Lanarkshire , Scotland , the son of Jean Ross ( née
Howie ) , a teacher and pianist , and Ian Baxter McMillan , a general practitioner who also served as a forensic police surgeon . He has an older sister , Annie
, and a younger sister , Jane . Coltrane is the great - grandson of Scottish businessman Thomas W. Howie and the nephew of businessman Forbes Howie . He was
educated at Glenalmond College , an independent school in Perthshire , from which he was nearly expelled after hanging the prefects ' gowns from the school
clocktower . Though he later described his experiences there as deeply unhappy , he played for the rugby First XV , was head of the school 's debating society
and won prizes for his art .
</item 7>

<item 8>
Robbie Coltrane
J.K. Rowling , author of the Harry Potter series , had Coltrane at the top of her list to play Hagrid and , when asked whom she would like to see in the role ,
responded `` Robbie Coltrane for Hagrid '' in one quick breath . There is also an unnamed 8ft actor who stands in for the 6ft 1in Coltrane in some scenes .
Coltrane has also presented a number of documentary programmes for the British ITV network based around his twin passions for travel and transportation .
Coltrane in a Cadillac ( 1993 ) saw him cross North America from Los Angeles to New York behind the wheel of a 1951 Cadillac Series 62 coupe convertible , a
journey of 3,765 miles ( 6,059 km ) which he completed in 32 days . In 1997 , Coltrane appeared in a series of six programmes under the title Coltrane 's Planes
and Automobiles , in which he extolled the virtues of the steam engine , the diesel engine , the supercharger , the V8 engine , the two - stroke engine , and
the jet engine .
</item 8>

<item 9>
Robbie Coltrane
Jump up ^ `` Robbie Coltrane 's magical career '' . BBC News . 31 December 2005 . Jump up ^ `` Robbie Coltrane biography '' . Tiscali.co.uk . Retrieved 15
August 2011 . Jump up ^ Paton , Maureen ( 20 March 2003 ) . `` ' Hagrid ? I 'm just Dad ' '' . London : Telegraph . Jump up ^ `` Scotland on Sunday '' .
Scotlandonsunday.scotsman.com . Retrieved 15 August 2011 . Jump up ^ `` Dr Edward `` Fitz '' Fitzgerald `` . Crackertv.co.uk . Retrieved 15 August 2011 . Jump
up ^ `` j.k. rowling '' . Neatorama . Retrieved 15 August 2011 . Jump up ^ Alderson , Andrew ( 11 November 2001 ) . `` ' They really do look as I 'd imagined
they would inside my head ' '' . The Sunday Telegraph . Accio-quote.org . Retrieved 15 August 2011 . ( Accio Quote !
</item 9>

<item 10>
Robbie Coltrane
Robbie Coltrane - wikipedia Robbie Coltrane Jump to : navigation , search Robbie Coltrane OBE Coltrane in April 2007 Anthony Robert McMillan ( 1950 - 03 - 30 )
30 March 1950 ( age 67 ) Rutherglen , South Lanarkshire , Scotland Nationality British Alma mater University of Edinburgh Occupation Actor , Author Years active
1979 -- present Height 6 ft 1 in ( 185 cm ) Spouse ( s ) Rhona Gemmell ( m . 1999 -- 2003 ) Children Robbie Coltrane , OBE ( born Anthony Robert McMillan ; 30
March 1950 ) is a Scottish actor and author . He is known for his roles as Rubeus Hagrid in the Harry Potter films , as Valentin Dmitrovich Zukovsky in the
James Bond films GoldenEye and The World Is Not Enough , and as Dr. Eddie `` Fitz '' Fitzgerald in the British TV series Cracker during the 1990s .
</item 10>
=== END ITEM id=b7d739078ddebcba ===

=== ITEM id=65eb3ed1c1fa5937 ===
Gold sentences and facts:
Sentence 0: The Curse of Oak Island follows brothers Marty and Rick Lagina , originally from Kingsford , Michigan , through their efforts to find the speculated treasure or historical artifacts believed to be on Oak Island . The series discusses the history of the island , recent discoveries , theories , and prior attempts to investigate the site . Areas of interest include the `` Money Pit '' , Borehole 10 - x , Smith 's Cove , `` Nolan 's Cross '' , the `` Hatch '' , the `` Watchtower '' and the `` Swamp '' .
  Fact 0.0: The Curse of Oak Island is a series.
  Fact 0.1: The Curse of Oak Island follows brothers Marty and Rick Lagina.
  Fact 0.2: Marty and Rick Lagina are originally from Kingsford, Michigan.
  Fact 0.3: The Curse of Oak Island follows the Laginas' efforts to find the speculated treasure or historical artifacts believed to be on Oak Island.
  Fact 0.4: The series discusses the history of Oak Island.
  Fact 0.5: The series discusses recent discoveries on Oak Island.
  Fact 0.6: The series discusses theories about Oak Island.
  Fact 0.7: The series discusses prior attempts to investigate the site on Oak Island.
  Fact 0.8: Areas of interest in the series include the "Money Pit".
  Fact 0.9: Areas of interest in the series include Borehole 10-x.
  Fact 0.10: Areas of interest in the series include Smith's Cove.
  Fact 0.11: Areas of interest in the series include "Nolan's Cross".
  Fact 0.12: Areas of interest in the series include the "Hatch".
  Fact 0.13: Areas of interest in the series include the "Watchtower".
  Fact 0.14: Areas of interest in the series include the "Swamp".

Retrieved items:
<item 1>
The Curse of Oak Island
overall No. in season Title Original air date Viewers ( millions ) 29 `` Going for Broke '' November 15 , 2016 ( 2016 - 11 - 15 ) 2.62 The team agrees to turn
focus over to the Money Pit and review what they have collected . Researcher Zena Halpern presents a handmade copy of what she claims is a French map from 1347
of Oak Island . The map shows locations for a `` hatch '' , a `` valve '' and an `` anchor '' on the Island . A theory is presented based on a purported
decoding of a cipher that Joab at the time of King David transported the Ark of the Covenant to North Africa from which it was later transported onward by the
Knights Templar from the then - Portuguese town of Safi in Morocco to Oak Island . 30 `` Always Forward '' November 22 , 2016 ( 2016 - 11 - 22 ) 2.66 While
preparing to excavate the Money Pit , the team investigates an 18th century well on mainland Nova Scotia .
</item 1>

<item 2>
The Curse of Oak Island
61 17 `` A Family Album '' February 27 , 2018 ( 2018 - 02 - 27 ) 1.98 The episode includes accounts of activities on the Oak Island by the McGinnis family ,
descendants of Samuel Ball , William Chappell 's family , the Restall Family and the family of Robert Dunfield . The Blankenship / Nolan disputes on Oak Island
are discussed . The episode ends with a short history of the Lagina brothers ' interest in Oak Island . Note : This episode is a special . 62 18 `` Amazing
Discoveries '' March 6 , 2018 ( 2018 - 03 - 06 ) 2.89 The gem in the Jewelry fragment found on the island is a Rhodolite Garnet stone of small value . The
drilling obstruction in the `` money pit '' borehole is determined by divers to be a granite boulder . References ( edit ) ^ Jump up to : `` Shows A-Z - curse
of oak island , the on history '' . The Futon Critic . Retrieved December 6 , 2017 .
</item 2>

<item 3>
The Curse of Oak Island
On August 2 , 2016 , it was announced that Nova Scotia Business Inc. approved $1,271,546 in film funding for the production of the 4th season . In October 2017
, a fifth season was announced , which began airing on November 7 , 2017 . Episodes ( edit ) Season 1 ( 2014 ) ( edit ) No . overall No. in season Title
Original air date Viewers ( millions ) `` What Lies Below '' January 5 , 2014 ( 2014 - 01 - 05 ) 2.53 Brothers Rick and Marty Lagina now own most of Oak Island
. Determined to resume the search for treasure on the island , they explore an abandoned shaft called Borehole 10 - X and use water pressure to pump material to
the surface . They find a tiny piece of metal .
</item 3>

<item 4>
The Curse of Oak Island
Lee Lamb and Richard Restall visit the island . Richard Restall is identified conclusively as the owner of the toy gun . 49 5 `` Bone Dry '' December 5 , 2017 (
2017 - 12 - 05 ) 3.16 An iron spike is found on the beach . Some small pottery fragments are brought up from one of the boreholes . Two human bone fragments
were recovered and sent in for testing . 50 6 `` Remains of the Day '' December 12 , 2017 ( 2017 - 12 - 12 ) 3.01 A piece of leather is found . Rick Lagina has
headaches and a rash . He is later diagnosed as having lyme disease . The team intends to have the previously discovered human bone fragments carbon dated . 51
7 `` The Lot Thickens '' December 19 , 2017 ( 2017 - 12 - 19 ) 3.13 A theory is presented that Sir Francis Drake is buried on Oak Island in a vault with gold
treasure . It is suggested that the lack of any records documenting Drake travelling to Canada is proof that he must have travelled to Canada .
</item 4>

<item 5>
The Curse of Oak Island
The Curse of Oak Island - wikipedia The Curse of Oak Island Jump to : navigation , search The Curse of Oak Island Genre Reality Narrated by Robert Clotworthy
Country of origin Canada Original language ( s ) English No. of seasons 5 No. of episodes 62 Production Executive producer ( s ) Kevin Burns Camera setup
Multiple Running time 43 minutes Production company ( s ) Prometheus Entertainment Distributor A&E Networks Release Original network History Canada Original
release January 5 , 2014 ( 2014 - 01 - 05 ) -- present External links Website Production website The Curse of Oak Island is an active reality television series
that first premiered in Canada on the History network on January 5 , 2014 . The show features what is known as the Oak Island mystery , showing efforts to
search for historical artifacts and treasure .
</item 5>

<item 6>
The Curse of Oak Island
S. Wild West Tech The XY Factor Upcoming Blue Book The Commanders Retrieved from ``
https://en.wikipedia.org/w/index.php?title=The_Curse_of_Oak_Island&oldid=832882458 '' Categories : 2014 American television series debuts History ( U.S. TV
channel ) original programs English - language television programs 2010s American television series Television shows set in Nova Scotia Oak Island Hidden
categories : Use mdy dates from August 2014 Talk Contents About Wikipedia Français Latviešu Română Русский Edit links This page was last edited on 28 March 2018
, at 14 : 10 . About Wikipedia
</item 6>

<item 7>
The Curse of Oak Island
2018 '' . Showbuzz Daily . Retrieved March 7 , 2018 . Further reading ( edit ) MacDonald , David ( 1965 ) . `` Treasure Hunt : The Mystery of Oak Island '' .
Reader 's Digest -- via readersdigest.ca .
</item 7>

<item 8>
The Curse of Oak Island
Series overview ( edit ) Season Episodes Originally aired First aired Last aired 5 January 5 , 2014 ( 2014 - 01 - 05 ) February 9 , 2014 ( 2014 - 02 - 09 ) 10
November 4 , 2014 ( 2014 - 11 - 04 ) January 13 , 2015 ( 2015 - 01 - 13 ) 13 November 10 , 2015 ( 2015 - 11 - 10 ) February 2 , 2016 ( 2016 - 02 - 02 ) 16
November 15 , 2016 ( 2016 - 11 - 15 ) February 21 , 2017 ( 2017 - 02 - 21 ) 5 18 November 7 , 2017 ( 2017 - 11 - 07 ) March 6 , 2018 ( 2018 - 03 - 06 )
Production ( edit ) Having started in 2014 , The Curse of Oak Island is currently in its fifth season . On August 2 , 2016 , it was announced that Nova Scotia
Business Inc.
</item 8>

<item 9>
The Curse of Oak Island
Contents ( hide ) 1 Overview 1.1 Background 1.2 Theories 2 Series overview 3 Production 4 Episodes 4.1 Season 1 ( 2014 ) 4.2 Season 2 ( 2014 -- 2015 ) 4.3
Season 3 ( 2015 -- 2016 ) 4.4 Season 4 ( 2016 -- 2017 ) 4.5 Season 5 ( 2017 -- 2018 ) 5 References 6 Further reading 7 External links Overview ( edit ) The
Curse of Oak Island follows brothers Marty and Rick Lagina , originally from Kingsford , Michigan , through their efforts to find the speculated treasure or
historical artifacts believed to be on Oak Island . The series discusses the history of the island , recent discoveries , theories , and prior attempts to
investigate the site . Areas of interest include the `` Money Pit '' , Borehole 10 - x , Smith 's Cove , `` Nolan 's Cross '' , the `` Hatch '' , the ``
Watchtower '' and the `` Swamp '' .
</item 9>

<item 10>
The Curse of Oak Island
Background ( edit ) The Lagina brothers became fascinated with the island after reading the January 1965 issue of Reader 's Digest magazine that features an
article on the Restall family 's work to investigate the so - called `` Money Pit '' . Marty and his brother Rick obtained a controlling interest in Oak Island
Tours , which reportedly owns most of the island . The brothers were later approached by Prometheus Entertainment to do a reality show . Rick and Marty have
engaged the assistance of father - and - son Dan and Dave Blankenship , permanent residents of the island who have likewise been searching for treasure since
the 1960s . Theories ( edit ) The series explores various Oak Island theories and conjectures through conversations with independent researchers .
</item 10>
=== END ITEM id=65eb3ed1c1fa5937 ===

=== ITEM id=fa13af32beb3a7c4 ===
Gold sentences and facts:
Sentence 0: John Ernest Crawford ( born March 26 , 1946 ) is an American character actor , singer , and musician . At age 12 , Crawford rose to fame for playing Mark McCain , the son of Lucas McCain ( played by Chuck Connors ) , in the popular ABC Western series , The Rifleman , which originally aired from 1958 to 1963 . Crawford first performed before a national audience as a Mouseketeer .
  Fact 0.0: John Ernest Crawford was born March 26, 1946.
  Fact 0.1: John Ernest Crawford is an American character actor.
  Fact 0.2: John Ernest Crawford is an American singer.
  Fact 0.3: John Ernest Crawford is an American musician.
  Fact 0.4: At age 12, John Ernest Crawford rose to fame for playing Mark McCain.
  Fact 0.5: Mark McCain is the son of Lucas McCain.
  Fact 0.6: Lucas McCain was played by Chuck Connors.
  Fact 0.7: Crawford played Mark McCain in the popular ABC Western series The Rifleman.
  Fact 0.8: The Rifleman originally aired from 1958 to 1963.
  Fact 0.9: John Ernest Crawford first performed before a national audience as a Mouseketeer.

Retrieved items:
<item 1>
Johnny Crawford
Johnny Crawford - wikipedia Johnny Crawford Jump to : navigation , search For the Canadian ice hockey player and coach , see Johnny Crawford ( ice hockey ) .
Johnny Crawford Johnny Crawford in The Rifleman ( 1961 ) John Ernest Crawford ( 1946 - 03 - 26 ) March 26 , 1946 ( age 72 ) Los Angeles , California , U.S.
Occupation Actor , singer , musician , band leader Years active 1954 -- 96 Spouse ( s ) Charlotte Crawford ( 1995 -- present ) ( separated ) Relatives Robert L.
Crawford , Jr. ( brother ) Website Johnny Crawford fansite Johnny Crawford and Chuck Connors in The Rifleman ( 1960 ) John Ernest Crawford ( born March 26 ,
1946 ) is an American character actor , singer , and musician . At age 12 , Crawford rose to fame for playing Mark McCain , the son of Lucas McCain ( played by
Chuck Connors ) , in the popular ABC Western series , The Rifleman , which originally aired from 1958 to 1963 . Crawford first performed before a national
audience as a Mouseketeer .
</item 1>

<item 2>
Johnny Crawford
The third pilot , which was made as an episode of Dick Powell 's Zane Grey Theater , was picked up by ABC and the first season of The Rifleman began filming in
July 1958 . Crawford was nominated for an Emmy Award , at age 13 , for his role as Mark McCain , the son of Lucas McCain , played by Chuck Connors , in the Four
Star Television series The Rifleman , which originally aired from 1958 to 1963 . Throughout The Rifleman 's five seasons , a remarkable on - screen chemistry
existed between Connors and Crawford in the depiction of their father - son relationship . They were still close friends when Connors died on November 10 , 1992
, and Crawford gave a eulogy at Connors ' memorial service . During the late 1950s and early 1960s , Crawford had wide popularity with American teenagers and a
recording career that generated four Billboard Top 40 hits , including the single , `` Cindy 's Birthday '' , which peaked at number 8 , in 1962 .
</item 2>

<item 3>
Graph query row:
entity: Mark McCain
relationship: RELATED_TO
relationship_detail: None
fact: None
related: Johnny Crawford
</item 3>

<item 4>
Johnny Crawford
His first important break as an actor followed with the title role in a Lux Video Theatre production of `` Little Boy Lost '' , a live NBC broadcast on March 15
, 1956 . He also appeared in the popular Western series The Lone Ranger , in 1956 , in one of the few color episodes of that series . Following that performance
, the young actor worked steadily with many seasoned actors and directors . Freelancing for 2 1 / 2 years , he accumulated almost 60 television credits ,
including featured roles in three episodes of NBC 's The Loretta Young Show and an appearance as Manuel in , `` I Am an American '' , an episode of the
syndicated crime drama Sheriff of Cochise . By the spring of 1958 , he had also performed 14 demanding roles in live teleplays for NBC 's Matinee Theatre ,
appeared on CBS 's sitcom , Mr. Adams and Eve , in the Wagon Train episode `` The Sally Potter Story '' ( in which Martin Milner also appeared ) and on the
syndicated series , Crossroads , Sheriff of Cochise , and Whirlybirds , and made three pilots of TV series .
</item 4>

<item 5>
Johnny Crawford
He played a young deputy Billy Norris , in The Big Valley episode `` The Other Face Of Justice '' in 1969 . While enlisted in the United States Army for two
years , Crawford worked on training films as a production coordinator , assistant director , script supervisor , and occasional actor . His rank was sergeant at
the time of his honorable discharge in December 1967 . In 1968 , Crawford played a soldier wanted for murder in `` By the Numbers '' , an episode of the popular
TV series Hawaii Five - O . The Resurrection of Broncho Billy was a USC student film Crawford agreed to do as a favor to his close friend , producer John
Longenecker . It won the 1970 Academy Award for Best Live Action Short Subject . The Naked Ape was a partially animated 1973 feature film starring Crawford and
Victoria Principal , produced by Hugh Hefner . In 1976 , Johnny Crawford co-starred as Ben Shelby in the tenth episode of season three of Little House on the
Prairie , `` The Hunters '' . In 1985 , Crawford appeared as Noah Paisley , sheriff 's deputy , in an episode of Murder , She Wrote .
</item 5>

<item 6>
The Dukes of Hazzard (film)
Gainey as Sheriff Rosco P. Coltrane Michael Weston as Deputy Enos Strate Lynda Carter as Pauline James Roday as Billy Prickett Kevin Heffernan as Derek `` Sheev
'' Sheevington Nikki Griffin as Katie - Lynn Johnson Jacqui Maxwell as Annette Alice Greczyn as Laurie Pullman Junior Brown as The Balladeer ( narrator ) Joe
Don Baker as Governor Jim Applewhite Barry Corbin as Bill Pullman Andrew Prine as Angry Man Randy Orton as Bo Duke ( Stunt double ) Cameos ( edit ) All five
members of the comedy film troupe Broken Lizard make appearances in the film , classified as cameos , except for Kevin Heffernan , who had a larger speaking
role ( Sheev ) . Broken Lizard cameos Steve Lemme appears as Jimmy Pullman , the son of Bill Pullman , in an early car chase scene in which he accidentally
shoots the inside of his father 's truck . Jay Chandrasekhar and Erik Stolhanske reprise their roles as Ramathorn and Rabbit from the Broken Lizard comedy ,
Super Troopers .
</item 6>

<item 7>
Johnny Crawford
p . A4 . Retrieved January 25 , 2014 . Jump up ^ `` Johnny Crawford ... ' the Son of Rifleman ' '' . TV Week Magazine . The Evening Independent . St. Petersburg
, Florida . May 24 , 1959 . p. 7 . Retrieved January 25 , 2014 . Jump up ^ `` Answers : Johnny Crawford A Mixture '' . The Evening Independent . St. Petersburg
, Florida . November 20 , 1963 . p . A9 . Retrieved January 25 , 2014 . His mother is from Belgium . Jump up ^ 1959 Emmys Jump up ^ `` '' A Very Bright Boy ''
on The Donna Reed Show , December 21 , 1961 `` . Internet Movie Data Base . Retrieved January 14 , 2013 . Jump up ^ `` Johnny Crawford : The Naked Ape '' . Mary
Anderson . Retrieved January 25 , 2014 . Jump up ^ `` Sweepin ' the Clouds Away '' , cduniverse.com ; accessed April 11 , 2015 .
</item 7>

<item 8>
Homestead Acts
Wheeler Outlaws Billy the Kid Black Bart `` Curly Bill '' Brocius Butch Cassidy Billy Clanton Ike Clanton Bill Dalton Bill Doolin John Wesley Hardin Johnny
Ringo Jesse James Tom Ketchum Frank McLaury Tom McLaury Joaquin Murrieta Cochise County Cowboys Belle Starr Soapy Smith Sundance Kid Cole Younger Soldiers and
scouts Frederick Russell Burnham Kit Carson `` Buffalo Bill '' Cody Texas Jack Omohundro James C. Cooney George Crook George Armstrong Custer Samuel P.
Heintzelman Tom Horn Calamity Jane Luther Kelly Ranald S. Mackenzie Charley Reynolds Philip Sheridan Al Sieber Others John Jacob Astor William H. Boring
Jonathan R. Davis George Flavel C.S. Fly John Joel Glanton George E. Goodfellow Doc Holliday Seth Kinman Nat Love Sylvester Mowry Emperor Norton Annie Oakley
Thomas William Sweeny Jack Swilling Native Americans Apache Arapaho Arikara Assiniboine ( Nakota ) Blackfoot Cahuilla Cayuse Cheyenne Chinook Chippewa ( Ojibwe
) Caddo Cocopah Comanche Crow Dakota Five Civilized Tribes Hidatsa
</item 8>

<item 9>
Rio Bravo (film)
In this film , Robert Mitchum played the Dean Martin role , Arthur Hunnicutt the Walter Brennan character , and James Caan the Ricky Nelson role . Hawks again
named the Nelson / Caan character after a state ( in this case , Mississippi ) and in a wry , humorous twist on the original film , Hawks made him inept with
firearms , but skilled with a knife . The second remake , Rio Lobo , was made in 1970 with a plot much further off the original mold , starting with the absence
of a lawman - turned - drunkard character . This began with a Confederate train robbery of a Union gold shipment during the American Civil War , then moved to a
postwar Texas county thoroughly controlled by a rich , arrogant rancher . The heroes , with the exception of an old man similar to Brennan and Hunnicutt 's
characters in the previous pictures ( Jack Elam here ) , were complete outsiders . Along with Wayne and Elam , this movie starred Mexican film star Jorge Rivero
( as Frenchie ) , Christopher Mitchum ( Robert Mitchum 's son ) , and Jennifer O'Neill .
</item 9>

<item 10>
Johnny Crawford
Jump up ^ `` Johnny Crawford '' . Scott Stander . Retrieved January 25 , 2014 . Jump up ^ `` Angry Woman -- Victoria Jackson on `` Tonight Show '' b / w `` .
Youtube . Retrieved January 25 , 2014 . Biography portal California portal Los Angeles portal Film portal Television portal Music portal United States Army
portal External links ( edit ) Johnny Crawford on The Rifleman Broncho Billy short film Johnny Crawford 's MySpace site Johnny Crawford on IMDb VIAF : 63364407
LCCN : no91003539 ISNI : 0000 0001 1526 4402 GND : 142142425 SUDOC : 079205798 BNF : cb11219552n ( data ) MusicBrainz : bf40b9eb - 068a - 4bdd - be46 -
ec0b5bd7a177 BNE : XX4832002 SNAC : w6sk5xbc Retrieved from `` https://en.wikipedia.org/w/index.
</item 10>
=== END ITEM id=fa13af32beb3a7c4 ===

=== ITEM id=73d2666391596495 ===
Gold sentences and facts:
Sentence 0: The Roosevelt Corollary was an addition to the Monroe Doctrine articulated by President Theodore Roosevelt in his State of the Union address in 1904 after the Venezuela Crisis of 1902 -- 03 . The corollary states that the United States will intervene in conflicts between European countries and Latin American countries to enforce legitimate claims of the European powers , rather than having the Europeans press their claims directly .
  Fact 0.0: The Roosevelt Corollary was an addition to the Monroe Doctrine.
  Fact 0.1: President Theodore Roosevelt articulated the Roosevelt Corollary.
  Fact 0.2: Theodore Roosevelt articulated the Roosevelt Corollary in his State of the Union address in 1904.
  Fact 0.3: The Roosevelt Corollary was articulated after the Venezuela Crisis of 1902-03.
  Fact 0.4: The Roosevelt Corollary states that the United States will intervene in conflicts between European countries and Latin American countries.
  Fact 0.5: Under the Roosevelt Corollary, the United States intervenes to enforce legitimate claims of the European powers.
  Fact 0.6: Under the Roosevelt Corollary, the United States intervenes rather than having the Europeans press their claims directly.

Retrieved items:
<item 1>
Truman Doctrine
Tajikistan Turkmenistan Uzbekistan East China Hong Kong Japan Macau Mongolia North Korea South Korea Taiwan South Afghanistan Bangladesh Bhutan India Maldives
Nepal Pakistan ( military relations ) Sri Lanka Southeast Brunei Burma Cambodia East Timor Indonesia Laos Malaysia Philippines Singapore Thailand Vietnam West
Armenia Azerbaijan Bahrain Cyprus Georgia Iran Iraq Israel military relations Jordan Kuwait Lebanon Oman Palestine Qatar Saudi Arabia Syria Turkey United Arab
Emirates Yemen Europe East Belarus Bulgaria Czech Republic Hungary Moldova Poland Romania Russia Slovakia Ukraine North Denmark Estonia Finland Iceland Ireland
Latvia Lithuania Norway Sweden United Kingdom South Andorra Albania Bosnia and Herzegovina Croatia Greece Italy Kosovo Macedonia Malta Montenegro Portugal San
Marino Serbia Slovenia Spain Vatican City West Austria Belgium France Germany Liechtenstein Luxembourg Monaco Netherlands Switzerland Oceania Australia Fiji
Kiribati Marshall Islands Micronesia Nauru New Zealand Palau Papua New Guinea Samoa Solomon Islands Tonga Tuvalu Vanuatu Former states Kingdom of Hawaii
Republic of Texas Russian Empire Soviet Union Multilateral relations Arab League European Union Latin America United Nations Third Border Initiative
International organizations ( hide ) Doctrines , policies , concepts Presidential doctrines Proclamation of Neutrality Monroe Roosevelt Corollary Good Neighbor
policy Truman Eisenhower Kennedy Johnson Nixon Carter Reagan Clinton Bush Obama Other doctrines Lodge Corollary Stimson Kirkpatrick Weinberger Powell
</item 1>

<item 2>
Roosevelt Corollary
This was used to justify Operation PBSUCCESS that deposed the democratically elected President Jacobo Árbenz and installed the military regime of Carlos
Castillo Armas , the first in a series of military dictators in the country . Criticism ( edit ) The argument made by Mitchener and Weidenmier in 2006 in
support of the Roosevelt Corollary to the Monroe Doctrine has been criticized on the grounds that it `` represent ( s ) the one - sided approach that some
scholars bring to the study of imperialistic and hegemonic interventions and also highlight how arguments for the general utility of imperialism are
increasingly made and accepted . '' Christopher Coyne and Stephen Davies argue that a foreign policy modeled on the Roosevelt Corollary leads to negative
consequences both in national security terms and in terms of its effect on domestic politics . Critics , such as Noam Chomsky , have argued that the Roosevelt
Corollary was merely a more explicit imperialist threat , building on the Monroe Doctrine , indicating that the U.S. would intervene not only in defense of
South American states in the face of European imperialism , but would also use its muscle to obtain concessions and privileges for American corporations .
</item 2>

<item 3>
Roosevelt Corollary
The Roosevelt Corollary was an addition to the Monroe Doctrine articulated by President Theodore Roosevelt in his State of the Union address in 1904 after the
Venezuela Crisis of 1902 -- 03 . The corollary states that the United States will intervene in conflicts between European countries and Latin American countries
to enforce legitimate claims of the European powers , rather than having the Europeans press their claims directly . Roosevelt tied his policy to the Monroe
Doctrine , and it was also consistent with his foreign policy included in his Big Stick Diplomacy . Roosevelt stated that in keeping with the Monroe Doctrine ,
the United States was justified in exercising `` international police power '' to put an end to chronic unrest or wrongdoing in the Western Hemisphere . While
the Monroe Doctrine had sought to prevent European intervention , the Roosevelt Corollary was used to justify US intervention throughout the hemisphere . In
1934 , President Franklin D. Roosevelt renounced interventionism and established his Good Neighbor policy for the Western Hemisphere .
</item 3>

<item 4>
Roosevelt Corollary
of Nations European Union Latin America United Nations Third Border Initiative International organizations ( hide ) Doctrines , policies , concepts Presidential
doctrines Proclamation of Neutrality Monroe Roosevelt Corollary Good Neighbor policy Truman Eisenhower Kennedy Johnson Nixon Carter Reagan Clinton Bush Obama
Other doctrines Lodge Corollary Stimson Kirkpatrick Weinberger Powell Rumsfeld Wolfowitz Policies and concepts Blowback Containment Domino theory
Non-interventionism Progressive realism Rollback Special Relationship Taiwan Relations Act United States intervention in Latin America Policy Monroe Doctrine (
1823 ) Platt Amendment ( 1901 -- 04 ) Roosevelt Corollary / Big Stick ideology ( 1904 ) Good Neighbor policy ( 1933 ) Dollar diplomacy Banana Wars Wars Mexican
-- American War ( 1846 - 48 ) Spanish -- American War ( 1898 ) Border War ( 1910 -- 19 ) United States involvement in the Mexican Revolution ( 1916 -- 19 )
Overt actions and occupations Paraguay expedition ( 1858 ) First Occupation of Cuba ( 1899 -- 1902 ) Panama -- Colombia separation and Hay -- Bunau - Varilla
Treaty ( 1903 ) Second Occupation of Cuba ( 1906 -- 09 ) Occupations of Honduras Occupation of Nicaragua (
</item 4>

<item 5>
Truman Doctrine
George F. Kennan ( 1904 - 2005 ) proposed the doctrine of containment in 1946 . In February 1946 , Kennan , an American diplomat in Moscow , sent his famed ``
Long Telegram '' , which predicted the Soviets would only respond to force and that the best way to handle them would be through a long - term strategy of
containment , that is stopping their geographical expansion . After the British warned that they could no longer help Greece , and following Prime Minister
Konstantinos Tsaldaris 's visit to Washington in December 1946 to ask for American assistance , the U.S. State Department formulated a plan . Aid would be given
to both Greece and Turkey , to help cool the long - standing rivalry between them . American policy makers recognized the instability of the region , fearing
that if Greece was lost to communism , Turkey would not last long . Similarly , if Turkey yielded to Soviet demands , the position of Greece would be endangered
. A regional domino effect threat therefore guided the American decision .
</item 5>

<item 6>
Roosevelt Corollary
Serge Ricard of the University of Paris goes even further , stating that the Roosevelt Corollary was not merely an addendum to the earlier Monroe Doctrine ,
through which the U.S. pledged to protect the Americas from European imperialist interventions . Rather , the Roosevelt Corollary was `` an entirely new
diplomatic tenet that epitomized his ' big stick ' approach to foreign policy . '' In other words , while the Monroe Doctrine sought to bar entry to the
European empires , the Roosevelt Corollary announced the United States intention to take their place . See also ( edit ) International relations portal
Government of the United States portal United States portal Wikisource has original text related to this article : Roosevelt Corollary History of the United
States ( 1865 -- 1918 ) United States territorial acquisitions New Imperialism Dollar Diplomacy Big Stick ideology Drago Doctrine Bush Doctrine References (
edit ) Jump up ^ Theodore Roosevelt , 26th President . Prod . David Grubin . By David Grubin and Geoffrey C. Ward . Perf . Walter LaFeber . David Grubin
Productions , Inc. , 1996 .
</item 6>

<item 7>
Roosevelt Corollary
Contents ( hide ) 1 Use 1.1 Shift to the `` Good Neighbor '' policy 2 Criticism 3 See also 4 References 5 Bibliography Use ( edit ) See also : Overseas
interventions of the United States The Roosevelt Corollary was an addition to the Monroe Doctrine ; however , it could be seen as a departure . While the Monroe
Doctrine said European countries should stay out of Latin America , the Roosevelt Corollary took this further to say he had the right to exercise military force
in Latin American countries to keep European countries out . Historian Walter LaFeber wrote ( Roosevelt ) essentially turns the Monroe Doctrine on its head and
toes says the Europeans should stay out , but the United States has the right , under the doctrine , to go in to exercise police power to keep the Europeans out
of the way .
</item 7>

<item 8>
Roosevelt Corollary
In 1934 , Franklin D. Roosevelt further renounced interventionism and established his `` Good Neighbor policy '' that led to the annulment of the Platt
Amendment by the Treaty of Relations with Cuba in 1934 , and the negotiation of compensation for Mexico 's nationalization of foreign - owned oil assets in 1938
. Indeed , leaving unchallenged the emergence of dictatorships like that of Fulgencio Batista in Cuba , Rafael Leonidas Trujillo in the Dominican Republic ,
Anastasio Somoza in Nicaragua , and François Duvalier in Haiti were each considered to be `` Frankenstein dictators '' due to the mishandlings of the American
occupations in the countries . The era of the Good Neighbor Policy ended with the ramp - up of the Cold War in 1945 , as the United States felt there was a
greater need to protect the western hemisphere from Soviet influence . In 1954 , Secretary of State John Foster Dulles invoked the Monroe Doctrine and the
Roosevelt Corollary at the Tenth Pan-American Conference in Caracas , denouncing the intervention of Soviet Communism in Guatemala .
</item 8>

<item 9>
Roosevelt Corollary
It is a very nice twist on the Monroe Doctrine , and of course , it becomes very , very important because over the next 15 to 20 years , the United States will
move into Latin America about a dozen times with military force , to the point where the United States Marines become known in the area as `` State Department
Troops '' because they are always moving in to protect State Department interests and State Department policy in the Caribbean . So what Roosevelt does here ,
by redefining the Monroe Doctrine , turns out to be very historical , and it leads the United States into a period of confrontation with peoples in the
Caribbean and Central America , that was an imperative part of American imperialism . Roosevelt first used the Corollary to act in the Dominican Republic in
1904 , which at the time was severely indebted and becoming a failed state . The United States dispatched two warships and demanded the customs house be turned
over to U.S. negotiators , who then used a percentage of the proceeds to pay foreign creditors . This model -- in which United States advisors worked to
stabilize Latin American nations through temporary protectorates , staving off European action -- became known as `` dollar diplomacy '' .
</item 9>

<item 10>
Roosevelt Corollary
The Dominican experiment , like most other `` dollar diplomacy '' arrangements , proved temporary and untenable , and the United States launched a larger
military intervention in 1912 . U.S. Presidents also cited the Roosevelt Corollary as justification for U.S. intervention in Cuba ( 1906 -- 1909 ) , Nicaragua (
1909 -- 1910 , 1912 -- 1925 and 1926 -- 1933 ) , Haiti ( 1915 -- 1934 ) , and the Dominican Republic ( 1916 -- 1924 ) . Shift to the `` Good Neighbor '' policy
( edit ) Main article : Good Neighbor policy In 1928 , under President Calvin Coolidge , the Clark Memorandum -- often seen as a partial repudiation of the
Roosevelt Corollary -- stated that the U.S. did not have the right to intervene when there was a threat by European powers . Herbert Hoover also helped move the
U.S. away from the imperialist tendencies of the Roosevelt Corollary by going on good - will tours , withdrawing troops from Nicaragua and Haiti , and
abstaining from intervening in the internal affairs of neighboring countries .
</item 10>
=== END ITEM id=73d2666391596495 ===

=== ITEM id=4168543b7077fe41 ===
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
As of 2008 , Westwood One was no longer controlled by CBS , but the network retained its NFL broadcast rights . In 2011 , Westwood One was purchased by Dial
Global . Then in 2014 , Dial Global , including Westwood One , was acquired by Cumulus Media . The Spanish - language broadcast is carried on ESPN Deportes
Radio . Nielsen ratings ( edit ) This article needs to be updated . Please update this article to reflect recent events or newly available information . (
December 2016 ) The highest - rated Monday Night Football telecast on ABC was the Miami Dolphins ' victory over the previously undefeated Chicago Bears on
December 2 , 1985 , which drew a national Nielsen rating of 29.6 and a share of 46 . ABC 's lowest - rated MNF game was the St. Louis Rams ' defeat of the Tampa
Bay Buccaneers on October 18 , 2004 , which drew a 7.7 rating . During the 1995 -- 1996 television season , Monday Night Football averaged a 17.1 household
rating . Its main competitor , Murphy Brown on CBS , averaged a 12.3 rating .
</item 2>

<item 3>
Monday Night Football
1987 -- 2005 ) NFL Draft NFL on ABC Sports Football Sunday on ESPN Radio Studio programs The Blitz Monday Night Countdown NFL Matchup NFL Primetime Sunday NFL
Countdown NFL Live NFL Insiders Other football programs Arena Football League on ESPN CFL on TSN ESPN College Football USFL on ESPN Related articles History of
Monday Night Football NFL on television ( history ) Commentators Sean McDonough Play - by - play TBA Color commentary Lisa Salters Sideline reporter Former
commentators Bonnie Bernstein Mike Ditka Jon Gruden Ron Jaworski Suzy Kolber Tony Kornheiser Michele Tafoya Joe Theismann Mike Tirico Dick Vermeil Steve Young
Lore televised by ESPN ESPN takes over MNF 2005 Saints -- Giants / Cowboys -- Redskins Doubleheader 2006 Bears -- Cardinals Comeback Fail Mary Sunday Night
Football era American Bowl ( 1990 - 2005 ) Fútbol Americano Pro Bowls broadcast on ESPN 1988 1990 1991 1992 1994 2005 2006 2015 2016 2017 2018 Music Edd
Kalehoff Johnny Pearson Hank Williams , Jr . Songs `` All My Rowdy Friends Are Here on Monday Night '' `` Heavy
</item 3>

<item 4>
Monday Night Football
At the end of the song , Williams Jr. touched a foot pump , which supposedly contained the helmets of that night 's participating teams . The helmets were
launched from the home toward the stadium at which the game was held . Through computer - generated imagery , the helmets `` land '' at midfield during a live
shot , and then explode . The `` exploding helmets '' gimmick was also used at various times in the 1980s and 1990s during the pre-game tease . Williams Jr.
then appeared again at the end of the game to promote the next week 's matchup . ESPN also continued to promote upcoming albums through its use in bumper music
. On September 29 ( Baltimore Ravens at Pittsburgh Steelers ) , ESPN used `` Another Way to Die '' , a duet between Alicia Keys and Jack White of the White
Stripes -- the song was part of the soundtrack for the 2008 film Quantum of Solace , then the latest in the James Bond series . Monday Night Football celebrated
its 600th game broadcast on October 20 , 2008 in a game which the New England Patriots defeated the Denver Broncos , 41 -- 7 .
</item 4>

<item 5>
Monday Night Football
Numbers ( retired ) Hall of Fame Lore Nicknames AFL Merger International Foreign players London Toronto Bills Series List of games played outside the U.S. Media
TV NFL Network Radio Management Council Competition Committee Collective Bargaining Agreement NFLPA Player conduct Suspensions Player misconduct Combine Draft
Training camp Rivalries NFL Foundation Tied games Canceled games Lockouts Controversies Cheerleading Mascots Color Rush NFL on ABC Related programs Monday Night
Football ESPN Sunday Night Football American Football League on ABC Pro Football Highlights Non-NFL programs College Football on ABC USFL on ABC World League of
American Football on ABC Related articles History of Monday Night Football NFL on television ( history ) Monday Night Mayhem ( film ) ABC Monday Night Football
( video game ) Super Bowl TV ratings ( lead - out programs ) Chicago Bears ( home games ) 1953 1954 1955 Chicago Cardinals ( home games ) 1953 1954 1955 Los
Angeles Rams ( Pacific Time Zone affiliates ) 1955 San Francisco 49ers ( Pacific Time Zone affiliates ) 1955 Washington Redskins ( home games ) 1954
Commentators AFL All - Star Game American Bowl Pro Bowl World Bowl Postseason commentators AFL Championship Game NFL Championship Game Super Bowl Lore televised
by ABC `` Body Bag
</item 5>

<item 6>
Super Bowl XXXVII
League on ABC Pro Football Highlights Non-NFL programs College Football on ABC USFL on ABC World League of American Football on ABC Related articles History of
Monday Night Football NFL on television ( history ) Monday Night Mayhem ( film ) ABC Monday Night Football ( video game ) Super Bowl TV ratings ( lead - out
programs ) Chicago Bears ( home games ) 1953 1954 1955 Chicago Cardinals ( home games ) 1953 1954 1955 Los Angeles Rams ( Pacific Time Zone affiliates ) 1955
San Francisco 49ers ( Pacific Time Zone affiliates ) 1955 Washington Redskins ( home games ) 1954 Commentators AFL All - Star Game American Bowl Pro Bowl World
Bowl Postseason commentators AFL Championship Game NFL Championship Game Super Bowl Lore televised by ABC `` Body Bag Game '' Death of John Lennon Brett Favre
wins one for his father `` Monday Night Miracle '' Montana outduels Elway `` Snowball Game '' Joe Theismann 's career - ending sack `` The Night That Courage
Wore Orange '' World Bowl ' 91 Postseason lore `` Music City Miracle '' `` One Yard Short '' `` Wide Right '' Music Charles Fox Edd Kalehoff Johnny Pearson Hank
Williams , Jr .
</item 6>

<item 7>
NBC Sunday Night Football
On November 14 , the NFL decided to keep the Week 12 matchup between the Pittsburgh Steelers and Kansas City Chiefs on November 27 in place after the league
considered flexing it out for other matchups , particularly the AFC East showdown between the Buffalo Bills and the New York Jets and the interconference
matchup between the Tampa Bay Buccaneers and the Tennessee Titans , due to the AFC West ( of which the Chiefs are a member ) being a weak division for 2011 . On
December 7 , the NFL ended up keeping the Week 15 matchup between the Baltimore Ravens and the San Diego Chargers on December 18 , a decision that came a day
late due to the NFL Committees meetings that took place on the day before . NBC wanted the game between the New England Patriots and the Denver Broncos as it
featured a matchup between Tom Brady and Tim Tebow , two players with high popularity . While CBS did not protect that game , the network was fighting to keep
the game since they had lost the aforementioned Week 13 Broncos - Vikings game to Fox , denying the network the earlier chance to capitalize on Tebow 's
marketability .
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
NBC Sunday Night Football
Cavalcade of Sports Hambletonian The ' Lights MLB Game of the Week MLB on NBC Major League Baseball : An Inside Look MLS on NBC NBA on NBC NBA Showtime NBC
College Football Game of the Week Sportsworld Parent : NBCUniversal NFL on NBC Related programs The NFL on NBC pregame show ( Football Night in America ) NBC
Sunday Night Football NFL on NBC Radio Thursday Night Football ( 2016 -- present ) College football programs College Football on NBC ( Notre Dame ) College
Football on USA Other pro football programs Arena Football League on NBC Canadian Football League World League of American Football on USA XFL on NBC Related
articles NFL on television ( history ) Primary television stations Super Bowl TV ratings ( lead - out programs ) American Football League 1965 1966 1967 1968
1969 Baltimore Colts 1960 1961 Pittsburgh Steelers 1960 1961 Prime - time results Monday night NFL games prior to 1970 Sunday Night Football results ( 2006 -
present ) Commentators AFC Championship Game Commentator pairings Pro Bowl Pregame show panelists Super Bowl Pre-AFL -- NFL merger AFL Championship Game AFL All
- Star Game NFL Championship Game Preseason games American Bowl Lore 1982 CFL season
</item 9>

<item 10>
Monday Night Football
go.com/nfl/mnf ESPN Monday Night Football ( abbreviated as MNF and also known as ESPN Monday Night Football on ABC for rare live special broadcast ) is a live
television broadcast of weekly National Football League ( NFL ) games on ESPN in the United States . From 1970 to 2005 , it aired on sister broadcast network
ABC . Monday Night Football was , along with Hallmark Hall of Fame and the Walt Disney anthology television series , one of the longest - running prime time
programs ever on commercial network television , and one of the highest - rated , particularly among male viewers . MNF is preceded on air by Monday Night
Countdown served by Applebee 's . Monday Night Football is also broadcast in Canada on TSN and RDS , and in most of Europe . On September 7 , 2013 , the NFL
announced that British Eurosport would show Monday Night Football games live in the United Kingdom for the 2013 and 2014 seasons . The telecasts are also seen
in most of Australia on ESPN Australia , in Portugal on SportTV 3 and SportTV HD and on TV 2 Sport in Denmark , and in some other regions of the world outside
the U.S.
</item 10>
=== END ITEM id=4168543b7077fe41 ===
