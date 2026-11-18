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

=== ITEM id=059e084a9a24fe34 ===
Gold sentences and facts:
Sentence 0: Stone tools were made from a variety of stones . For example , flint and chert were shaped ( or chipped ) for use as cutting tools and weapons , while basalt and sandstone were used for ground stone tools , such as quern - stones . Wood , bone , shell , antler ( deer ) and other materials were widely used , as well . During the most recent part of the period , sediments ( such as clay ) were used to make pottery . Agriculture was developed and certain animals were domesticated as well .
  Fact 0.0: Stone tools were made from a variety of stones.
  Fact 0.1: Flint and chert were shaped, or chipped, for use as cutting tools and weapons.
  Fact 0.2: Basalt and sandstone were used for ground stone tools.
  Fact 0.3: Basalt and sandstone were used for ground stone tools such as quern-stones.
  Fact 0.4: Wood, bone, shell, and antler (deer) were widely used materials.
  Fact 0.5: Other materials were widely used as well.
  Fact 0.6: During the most recent part of the period, sediments such as clay were used to make pottery.
  Fact 0.7: Agriculture was developed.
  Fact 0.8: Certain animals were domesticated.

Retrieved items:
<item 1>
Stone Age
1.3 Acheulean in Africa 3.1. 1.4 Acheulean out of Africa 3.1. 2 Middle Paleolithic 3.1. 3 Upper Paleolithic 3.1. 4 Epipaleolithic / Mesolithic 3.1. 5 Neolithic
3.2 Three - stage chronology 3.2. 1 The Earlier or Early Stone Age ( ESA ) 3.2. 2 The Middle Stone Age ( MSA ) 3.2. 3 The Later Stone Age ( LSA ) 4 Material
culture 4.1 Tools 4.2 Food and drink 4.3 Shelter and habitat 4.4 Art 4.4. 1 Petroglyphs 4.4.
</item 1>

<item 2>
Stone Age
The Early Stone Age therefore is contemporaneous with the Lower Paleolithic and happens to include the same main technologies , Oldowan and Acheulean , which
produced Mode 1 and Mode 2 stone tools respectively . A distinct regional term is warranted , however , by the location and chronology of the sites and the
exact typology . The Middle Stone Age ( msa ) Main article : Middle Stone Age The Middle Stone Age was a period of African prehistory between Early Stone Age
and Late Stone Age . It began around 300,000 years ago and ended around 50,000 years ago . It is considered as an equivalent of European Middle Paleolithic . It
is associated with anatomically modern or almost modern Homo sapiens . Early physical evidence comes from Omo and Herto , both in Ethiopia and dated
respectively at c. 195 ka and at c. 160 ka . The Later Stone Age ( LSA ) Main article : Later Stone Age The Later Stone Age ( LSA , sometimes also called the
Late Stone Age ) refers to a period in African prehistory . Its beginnings are roughly contemporaneous with the European Upper Paleolithic .
</item 2>

<item 3>
Stone Age
Stone Age - wikipedia Stone Age Jump to : navigation , search For other uses , see Stone Age ( disambiguation ) . Ġgantija temples in Gozo , Malta . Some of the
world 's oldest free - standing structures . The Stone Age ↑ before Homo ( Pliocene ) Paleolithic Lower Paleolithic Late Stone Age Homo Control of fire Stone
tools Middle Paleolithic Middle Stone Age Homo neanderthalensis Homo sapiens Recent African origin of modern humans Upper Paleolithic Late Stone Age Behavioral
modernity , Atlatl , Origin of the domestic dog Epipaleolithic Mesolithic Microliths , Bow , Canoe Natufian Khiamian Tahunian Heavy Neolithic Shepherd Neolithic
Trihedral Neolithic Pre-Pottery Neolithic Neolithic Neolithic Revolution , Domestication Pottery Neolithic Pottery ↓ Chalcolithic The Stone Age was a broad
prehistoric period during which stone was widely used to make implements with an edge , a point , or a percussion surface . The period lasted roughly 3.4
million years and ended between 8700 BCE and 2000 BCE with the advent of metalworking .
</item 3>

<item 4>
Stone Age
- stone Storage pit Hunting Arrow Boomerang throwing stick Bow and arrow history Nets Spear Spear - thrower baton harpoon woomera Schöningen Spears Projectile
points Arrowhead Bare Island Cascade Clovis Cresswell Cumberland Eden Folsom Lamoka Manis Site Plano Transverse arrowhead Systems Game drive system Buffalo jump
Toolmaking Earliest toolmaking Oldowan Acheulean Mousterian Clovis culture Cupstone Fire hardening Gravettian culture Hafting Hand axe Grooves Langdale axe
industry Levallois technique Lithic core Lithic reduction analysis debitage flake Lithic technology Magdalenian culture Metallurgy Microblade technology Mining
Prepared - core technique Solutrean industry Striking platform Tool stone Uniface Yubetsu technique Other tools Adze Awl bone Axe Bannerstone Blade prismatic
Bone tool Bow drill Burin Canoe Oar Pesse canoe Chopper tool Cleaver Denticulate tool Fire plough Fire - saw Hammerstone Knife Microlith Quern - stone Racloir
Rope Scraper side Stone tool Tally stick Weapons Wheel illustration Architecture Ceremonial Göbekli Tepe Kiva Standing stones megalith row Stonehenge Pyramid
Dwellings Neolithic architecture British
</item 4>

<item 5>
Stone Age
The species who made the Pliocene tools remains unknown . Fragments of Australopithecus garhi , Australopithecus aethiopicus and Homo , possibly Homo habilis ,
have been found in sites near the age of the Gona tools . End of the Stone Age Innovation of the technique of smelting ore ended the Stone Age and began the
Bronze Age . The first most significant metal manufactured was bronze , an alloy of copper and tin , each of which was smelted separately . The transition from
the Stone Age to the Bronze Age was a period during which modern people could smelt copper , but did not yet manufacture bronze , a time known as the Copper Age
, or more technically the Chalcolithic , `` copper - stone '' age . The Chalcolithic by convention is the initial period of the Bronze Age . The Bronze Age was
followed by the Iron Age . The transition out of the Stone Age occurred between 6000 BCE and 2500 BCE for much of humanity living in North Africa and Eurasia .
</item 5>

<item 6>
Stone Age
Wikisource has the text of the 1920 Encyclopedia Americana article Stone Age . Giusepi , Robert A. ( 2000 ) . `` The Stone Age '' . History World International
. Retrieved 22 February 2011 . Kowalski , D.R. `` Stone Age Hand - axes '' . AerobiologicalEngineering.com . Retrieved 22 February 2011 . Kowalski , D.R. ``
Stone Age Habitats '' . AerobiologicalEngineering.com . Retrieved 22 February 2011 . `` PanAfrican Archaeological Association '' . Retrieved 28 February 2011 .
`` Society of Africanist Archaeologists '' . Retrieved 3 March 2011 . `` The ASA '' . Association of Social Anthropologists of the UK and Commonwealth . Human
Timeline ( Interactive ) -- Smithsonian , National Museum of Natural History ( August 2016 ) . Three - age system Stone Age Prehistoric technology Stone tool
Flint tool Paleolithic Lower Paleolithic Middle Paleolithic Upper Paleolithic Mesolithic Neolithic Japanese Paleolithic Paleolithic Europe Middle Stone Age
Later Stone Age Epipaleolithic
</item 6>

<item 7>
Stone Age
The Neolithic was characterized primarily by herding societies rather than large agricultural societies , and although there was copper metallurgy in Africa as
well as bronze smelting , archaeologists do not currently recognize a separate Copper Age or Bronze Age . Moreover , the technologies included in those ' stages
' , as Goodwin called them , were not exactly the same . Since then , the original relative terms have become identified with the technologies of the
Paleolithic and Mesolithic , so that they are no longer relative . Moreover , there has been a tendency to drop the comparative degree in favor of the positive
: resulting in two sets of Early , Middle and Late Stone Ages of quite different content and chronologies . By voluntary agreement , archaeologists respect the
decisions of the Pan-African Congress of Prehistory , which meets every four years to resolve archaeological business brought before it . Delegates are actually
international ; the organization takes its name from the topic . Louis Leakey hosted the first one in Nairobi in 1947 . It adopted Goodwin and Lowe 's 3 - stage
system at that time , the stages to be called Early , Middle and Later .
</item 7>

<item 8>
Stone Age
There are in effect two Stone Ages , one part of the Three - age and the other constituting the Three - stage . They refer to one and the same artifacts and the
same technologies , but vary by locality and time . The three - stage system was proposed in 1929 by Astley John Hilary Goodwin , a professional archaeologist ,
and Clarence van Riet Lowe , a civil engineer and amateur archaeologist , in an article titled `` Stone Age Cultures of South Africa '' in the journal Annals of
the South African Museum . By then , the dates of the Early Stone Age , or Paleolithic , and Late Stone Age , or Neolithic ( neo = new ) , were fairly solid and
were regarded by Goodwin as absolute . He therefore proposed a relative chronology of periods with floating dates , to be called the Earlier and Later Stone Age
. The Middle Stone Age would not change its name , but it would not mean Mesolithic . The duo thus reinvented the Stone Age . In Sub-Saharan Africa , however ,
iron - working technologies were either invented independently or came across the Sahara from the north ( see iron metallurgy in Africa ) .
</item 8>

<item 9>
Stone Age
Mesolithic Neolithic Japanese Paleolithic Paleolithic Europe Middle Stone Age Later Stone Age Epipaleolithic Pre-Pottery Neolithic Pre-Pottery Neolithic A
Pre-Pottery Neolithic B Neolithic Revolution List of Neolithic cultures of China Nordic Stone Age Neolithic British Isles South Asian Stone Age Stone - Age
Poland Recent African origin of modern humans Bronze Age Chalcolithic Bronze Atlantic Bronze Age Nordic Bronze Age Bronze Age in Romania Bronze Age collapse
Iron Age History of ferrous metallurgy Iron meteorite Metallurgy British Iron Age Pre-Roman Iron Age Roman Iron Age Germanic Iron Age Iron Age Scandinavia Iron
Age India Iron Age China Tsebelda culture List of archaeological periods List of time periods ( hide ) Prehistoric technology Prehistory timeline outline Stone
Age subdivisions New Stone Age synoptic table Technology history Tools Farming Neolithic Revolution founder crops New World crops Ard / plough Celt Digging
stick Domestication Goad Irrigation Secondary products Sickle Terracing Food processing Fire Basket Cooking Earth oven Granaries Grinding slab Ground stone
Hearth Aşıklı Höyük Qesem Cave Manos Metate Mortar and pestle Pottery Quern - stone Storage pit Hunting Arrow Boomerang throwing stick Bow and arrow history
Nets Spear Spear -
</item 9>

<item 10>
Stone Age
Its beginnings are roughly contemporaneous with the European Upper Paleolithic . It lasts until historical times and this includes cultures corresponding to
Mesolithic and Neolithic in other regions . Material culture Tools Stone tools were made from a variety of stones . For example , flint and chert were shaped (
or chipped ) for use as cutting tools and weapons , while basalt and sandstone were used for ground stone tools , such as quern - stones . Wood , bone , shell ,
antler ( deer ) and other materials were widely used , as well . During the most recent part of the period , sediments ( such as clay ) were used to make
pottery . Agriculture was developed and certain animals were domesticated as well . Some species of non-primates are able to use stone tools , such as the sea
otter , which breaks abalone shells with them . Primates can both use and manufacture stone tools . This combination of abilities is more marked in apes and men
, but only men , or more generally Hominans , depend on tool use for survival .
</item 10>
=== END ITEM id=059e084a9a24fe34 ===

=== ITEM id=b824eccc16772031 ===
Gold sentences and facts:
Sentence 0: `` Like a Boss '' is a song written and recorded by comedy hip hop troupe The Lonely Island for their debut studio album Incredibad . The song is a parody of the first official single from Slim Thug 's first album Already Platinum , also named `` Like a Boss '' . The song had multiple versions , including an audio - only that did not have the celebrity interviewer and a clean version that was produced in both audio and video . In 2011 it was also subject to a lawsuit from the producers of the song who claimed they never received proper compensation for their work .
  Fact 0.0: Like a Boss is a song.
  Fact 0.1: Like a Boss was written and recorded by the comedy hip hop troupe The Lonely Island.
  Fact 0.2: The Lonely Island recorded Like a Boss for their debut studio album Incredibad.
  Fact 0.3: Like a Boss is a parody of the first official single from Slim Thug's first album Already Platinum.
  Fact 0.4: The first official single from Slim Thug's first album Already Platinum is also named Like a Boss.
  Fact 0.5: Like a Boss had multiple versions.
  Fact 0.6: One version of Like a Boss was audio-only and did not have the celebrity interviewer.
  Fact 0.7: Like a Boss had a clean version that was produced in both audio and video.
  Fact 0.8: In 2011 Like a Boss was subject to a lawsuit.
  Fact 0.9: The 2011 lawsuit over Like a Boss was from the producers of the song.
  Fact 0.10: The producers of Like a Boss claimed they never received proper compensation for their work.

Retrieved items:
<item 1>
Like a Boss
In 2011 it was also subject to a lawsuit from the producers of the song who claimed they never received proper compensation for their work . Contents ( hide ) 1
Song 2 Video 3 Popularity 4 Lawsuit 5 Censorship 6 References 7 External links Song ( edit ) `` Like a Boss '' premiered as a Saturday Night Live Digital Short
on April 4 , 2009 . In the song , Andy Samberg is a businessman , evidently in middle management , having a performance review . The reviewer asks him to
describe the things he does in an average day , and does so for the remainder of the song in his trademark aggressive style while repeating the lyric `` like a
boss '' after every activity . He first lists activities which are fairly mundane , such as communicating to bigger bosses in corporate headquarters , approving
memorandums , remembering birthdays , leading a workshop , and promoting synergy . However , the song takes a foul turn after he mentions his female subordinate
Deborah , whom he repeatedly pursues , rejecting him , causing him to `` swallow sadness '' , face a sexual harassment lawsuit and be passed up for a promotion
.
</item 1>

<item 2>
Like a Boss
Banks , however , claims that he has yet to see any sort of a royalty for the song . Another track from the duo was later featured on the band 's 2011 album
Turtleneck & Chain in the song `` Shy Ronnie 2 : Ronnie & Clyde . '' Censorship ( edit ) This section does not cite any sources . Please help improve this
section by adding citations to reliable sources . Unsourced material may be challenged and removed . ( January 2015 ) ( Learn how and when to remove this
template message ) On the NBC version of this digital short , the part where Andy has a gun in his mouth and whines , `` Oh fuck , man , I ca n't fuckin ' do it
. Shit ! '' had the shot of Andy with the gun in his mouth shortened and redubbed the line to , `` Oh crap , man , I ca n't effin ' do it . Ugh ! ! ''
References ( edit ) Jump up ^ Van Dril , Eric ( 5 April 2009 ) . `` ' SNL ' Digital Short : ' Like A Boss ' ( With Video ) '' .
</item 2>

<item 3>
Like a Boss
Abby Elliott and Bobby Moynihan can be seen in the video as background dancers . Popularity ( edit ) Prior to the release of the video on Saturday Night Live ,
the term `` Like a Boss '' was not often searched for or used in social context . Though it was indeed searched for prior to the release of the video , the term
clearly spiked at release and peaked on Google in April 2009 , hitting a 100 on the Google Trend chart . It has also become popular on social media sites such
as Twitter Tumblr , and Facebook , where it is a popular hashtag to cite a task completed , goal achieved or successful action that someone is proud of doing .
It has also led to parody videos such as `` Like a Bus '' , where a bus is shown flying through traffic and often bad weather conditions with no regard or
slowing for the bad conditions . On 22 October 2012 , `` Like A Boss '' surpassed 100 million views . The video was featured by The Huffington Post , and TV
Guide . Lawsuit ( edit ) In November 2011 a production team from St. Louis filed charges against Andy Samberg and Co. , NBC Universal and Universal Republic
Records .
</item 3>

<item 4>
Like a Boss
`` ' SNL ' Digital Short : ' Like A Boss ' ( With Video ) '' . The Ripcord . Archived from the original on 11 April 2009 . Retrieved 2014 - 06 - 21 . Jump up ^
`` Google Trends - websearch interest : like a boss - Worldwide , 2004 - present '' . Google Trends . Jump up ^ `` Twitter / search - # likeaboss '' .
Twitter.com . Jump up ^ `` like a boss on Tumblr '' . Tumblr.com . Jump up ^ `` # likeaboss '' . Facebook.com . Jump up ^ `` Like a BUS ! Playlist '' .
Youtube.com . Jump up ^ `` Like A Boss : SNL Digital Short ( VIDEO ) '' . The Huffington Post . 6 May 2009 . Retrieved 2014 - 06 - 23 . Jump up ^ DiNunno , Gina
. `` Today 's News : Our Take - Seth Rogen and Andy Samberg in SNL Digital Short `` Like a Boss '' `` . TV Guide.com . Retrieved 23 June 2014 .
</item 4>

<item 5>
Like a Boss
Like a Boss - wikipedia Like a Boss Jump to : navigation , search ( hide ) This article has multiple issues . Please help improve it or discuss these issues on
the talk page . ( Learn how and when to remove these template messages ) This article 's lead section does not adequately summarize key points of its contents .
Please consider expanding the lead to provide an accessible overview of all important aspects of the article . Please discuss this issue on the article 's talk
page . ( January 2015 ) This article needs additional citations for verification . Please help improve this article by adding citations to reliable sources .
Unsourced material may be challenged and removed .
</item 5>

<item 6>
Like a Boss
Unsourced material may be challenged and removed . ( February 2011 ) ( Learn how and when to remove this template message ) ( Learn how and when to remove this
template message ) `` Like a Boss '' Song by The Lonely Island featuring Seth Rogen from the album Incredibad Recorded 2008 Genre Comedy hip hop Length 1 : 46
Label Universal Republic Records Songwriter ( s ) Andy Samberg , Akiva Schaffer , Jorma Taccone , A. Banks Producer ( s ) Rick tha Rular Incredibad track
listing `` Shrooms ( skit ) '' ( 9 ) `` Like a Boss '' ( 10 ) `` We Like Sportz '' ( 11 ) `` Like a Boss '' is a song written and recorded by comedy hip hop
troupe The Lonely Island for their debut studio album Incredibad . The song is a parody of the first official single from Slim Thug 's first album Already
Platinum , also named `` Like a Boss '' . The song had multiple versions , including an audio - only that did not have the celebrity interviewer and a clean
version that was produced in both audio and video .
</item 6>

<item 7>
Like a Boss
Louis filed charges against Andy Samberg and Co. , NBC Universal and Universal Republic Records . At the root of the suit was the claim that Monique `` Foxx
StarStrukk '' Hines and Aleric `` Rick Tha Rular '' Banks submitted the audio track for `` Like a Boss '' along with the audio track that would become another
hit for The Lonely Island , `` Shy Ronnie '' . The duo claimed that , through a mutual connection in 2008 , they knew Samberg and submitted their tracks for
inclusion in the upcoming debut album , Incredibad . The Lonely Island then recorded their own lyrics on top of the audio track to create the songs . The duo
from StarStrukk productions claimed that they were never compensated properly for their work , both in money and recognition / notoriety . Banks claimed that
originally he signed an agreement giving him a 50 % ownership of the song , given his position as writer and producer . According to his lawsuit , he received a
response from Universal Music Group that gave him only a 25 % interest in `` Like a Boss . ''
</item 7>

<item 8>
Like a Boss
The video ends with the giant fish delivering the final words `` like a boss . '' Video ( edit ) The video features Seth Rogen , recording over the vocals done
by Akiva Schaffer on the single. The video adds additional visuals not included with the words of the song . It ends with Samberg 's tombstone being shown (
which has the epitaph `` He was da Best '' ) , and a skeptical Rogen trying to confirm some of the more bizarre events with Samberg . The `` bomb the Russians
'' part shows Samberg attacking St. Basil 's Cathedral . The video features brief cameo appearances from Saturday Night Live writer John Mulaney as the worker
that Samberg micromanages , Casey Wilson as the woman giving Samberg the memo , Kristen Wiig as Samberg 's office crush Debra , Jason Sudeikis as Samberg 's
boss and Bill Hader as the man on whom Samberg performs fellatio . It also features Jorma Taccone as the man giving him the lawsuit , and Akiva Schaffer selling
him the gun . Abby Elliott and Bobby Moynihan can be seen in the video as background dancers .
</item 8>

<item 9>
Like a Boss
TV Guide.com . Retrieved 23 June 2014 . Jump up ^ Kaufman , Gill . `` ' SNL , ' Andy Samberg Sued By Producers over ' Like A Boss ' - MTV '' . MTV News .
Retrieved 11 July 2014 . External links ( edit ) Beard , Lanford ( November 8 , 2011 ) . `` NBC , Andy Samberg sued for copyright infringement '' .
Entertainment Weekly . Retrieved November 8 , 2011 . Original video : `` Like a Boss '' on YouTube Censored video : `` Like a Boss - Clean Version '' on YouTube
Video and Song Lyrics : `` Like a Boss '' Lonely Island 's Song Page : `` Like a Boss '' ( hide ) The Lonely Island Andy Samberg Akiva Schaffer Jorma Taccone
Discography Studio albums Incredibad Turtleneck & Chain The Wack Album Popstar : Never Stop Never Stopping Singles `` Dick in a Box '' `` Jizz in My Pants '' ``
I 'm on a Boat '' `` I Just Had Sex '' `` The Creep '' `` We 're Back ! ''
</item 9>

<item 10>
Like a Boss
`` Jack Sparrow '' `` 3 - Way ( The Golden Rule ) '' `` YOLO '' `` Spring Break Anthem '' `` Hugs '' Other songs `` Lazy Sunday '' `` Like a Boss '' `` Cool
Guys Do n't Look at Explosions '' `` Everything Is Awesome '' Film Hot Rod ( 2007 ) Popstar : Never Stop Never Stopping ( 2016 ) Television SNL Digital Shorts
Party Over Here Book Retrieved from `` https://en.wikipedia.org/w/index.php?title=Like_a_Boss&oldid=816720124 '' Categories : The Lonely Island songs 2009
singles Saturday Night Live sketches Comedy rap songs Hidden categories : Wikipedia introduction cleanup from January 2015 All pages needing cleanup Articles
covered by WikiProject Wikify from January 2015 All articles covered by WikiProject Wikify Articles needing additional references from February 2011 All
articles needing additional references Articles with multiple maintenance issues Articles with hAudio microformats Articles needing additional references from
January 2015 Talk Contents About Wikipedia Nederlands Simple English Edit links This page was last edited on 23 December 2017 , at 06 : 20 . Text is available
under the Creative Commons Attribution - ShareAlike License ; additional terms may apply .
</item 10>
=== END ITEM id=b824eccc16772031 ===

=== ITEM id=374527481387b3f8 ===
Gold sentences and facts:
Sentence 0: The Ming dynasty also engaged in a thriving trade with both Europe and Japan . The amount of silver flowing into the Ming dynasty was estimated by Joseph Needham at 300 million taels , which is equivalent to more than 190 billion dollars in today 's money . In addition to silver , the Ming also imported many European firearms , in order to ensure the modernness of their weapons .
  Fact 0.0: The Ming dynasty engaged in a thriving trade with both Europe and Japan.
  Fact 0.1: Joseph Needham estimated the amount of silver flowing into the Ming dynasty at 300 million taels.
  Fact 0.2: 300 million taels is equivalent to more than 190 billion dollars in today's money.
  Fact 0.3: In addition to silver, the Ming dynasty imported many European firearms.
  Fact 0.4: The Ming dynasty imported European firearms in order to ensure the modernness of their weapons.

Retrieved items:
<item 1>
Economy of the Ming dynasty
Many of these markets appeared in the rural countryside , where goods were exchanged and bartered . A second type of market that developed in China was the
urban - rural type , in which rural goods were sold to urban dwellers . This was particular the case when landlords decided to reside in the cities , and use
income coming from rural land holding to facilitate exchange in the cities . Another way this type of market was used was professional merchants who bought
rural goods in large quantities . The third type of market was the `` national market '' which was developed during the Song dynasty but particularly enhanced
during the Ming . This market involved not only the exchange described above , but also products produced directly for the market . Unlike earlier dynasties ,
many Ming peasants were no longer producing only products they needed ; many of them produced products for the market , which then they sold at a profit . Trade
and investment ( edit ) Main article : Haijin In the early Ming , after the devastation of the war which expelled the Mongols , the Hongwu Emperor imposed
severe restrictions on trade ( the `` haijin '' ) .
</item 1>

<item 2>
Economy of the Ming dynasty
Believing that agriculture was the basis of the economy , Hongwu favored that industry over all else , including that of merchants . After his death , most of
his policies were reversed by his successors . By the late Ming , the state was losing power to the very merchants which Hongwu had wanted to restrict . The
Ming dynasty also engaged in a thriving trade with both Europe and Japan . The amount of silver flowing into the Ming dynasty was estimated by Joseph Needham at
300 million taels , which is equivalent to more than 190 billion dollars in today 's money . In addition to silver , the Ming also imported many European
firearms , in order to ensure the modernness of their weapons . Trade and commerce thrived in this liberalized economy , and was aided by the construction of
canals , roads , and bridges by the Ming government . The Ming saw the rise of several merchant clans such as the Huai and Jin clans , who disposed of large
amounts of wealth . The gentry and merchant classes started to fuse , and the merchants gained power at the expense of the state . Some merchants were reputed
to have a treasure of 30 million taels .
</item 2>

<item 3>
Economy of the Ming dynasty
Economy of the Ming dynasty - wikipedia Economy of the Ming dynasty Jump to : navigation , search This article needs additional citations for verification .
Please help improve this article by adding citations to reliable sources . Unsourced material may be challenged and removed . ( October 2010 ) ( Learn how and
when to remove this template message ) The Ming Empire around 1580 The economy of the Ming dynasty ( 1368 -- 1644 ) of China was the largest in the world during
that period . It is regarded as one of China 's three golden ages ( the other two being the Han and Song periods ) . The period was marked by the increasing
political influence of the merchants , the gradual weakening of imperial rule , and technological advances .
</item 3>

<item 4>
Economy of the Ming dynasty
However , a second silver contraction occurred in the mid-17th century when King Philip IV began enforcing laws limiting direct trade between Spanish South
America and China at about the same time the new Tokugawa shogunate in Japan restricted most of its foreign exports , cutting off Dutch and Portuguese access to
its silver . The dramatic spike in silver 's value in China made payment of taxes nearly impossible for most provinces . In extremis , the government even
resumed use of paper currency amid Li Zicheng 's rebellion . Manufactures ( edit ) Privatization ( edit ) Another key feature of the Ming manufacturing industry
was privatization . Unlike the Song , in which state - owned enterprises played a large role , the Ming reverted to the old laissez faire policies of the Han by
privatizing the salt and tea industries . By the middle of the Ming dynasty , powerful groups of wealthy merchants had replaced the state as the dominant movers
behind Chinese industry . Emergence of wage labor ( edit ) The Ming government abolished the mandatory forced labor by peasants used in early dynasties and
replaced it with wage labor . A new class of wage laborers sprung up where none had existed before .
</item 4>

<item 5>
Economy of the Ming dynasty
The Columbian exchange brought crops such as corn with these foreign crops . During the Ming , specialized areas also popped up planting large numbers of cash
crops that could be sold at markets . Large numbers of peasants abandoned the land to become artisans . The population of the Ming boomed ; estimates for the
population of the Ming range from 160 to 200 million . Rural markets during the Ming ( edit ) Ming agriculture was much changed from the earlier areas ; firstly
, gigantic areas , devoting and specializing in cash crops , sprung up to demand from the new market economy . Secondly , agricultural tools and carts , some
water - powered , help to create a gigantic agricultural surplus which formed the basis of the rural economy . Besides rice , other crops were grown on a large
scale . Although images of autarkic farmers who had no connection to the rest of China may have some merit for the earlier Han and T'ang dynasties , this was
certainly not the case for the Ming dynasty . During the Ming dynasty , the increase in population and the decrease in quality land made it necessary that
farmers make a living off cash crops .
</item 5>

<item 6>
Economy of the Ming dynasty
Contents ( hide ) 1 Currency 2 Manufactures 2.1 Privatization 2.2 Emergence of wage labor 3 Early encouragement of agriculture under Hongwu 3.1 Emergence of
commercial plantations 3.2 Rural markets during the Ming 4 Trade and investment 4.1 Taxation 4.2 Weakening of the state 4.3 Sprouts of capitalism 5 See also 6
References 7 External links Currency ( edit ) Further information : Ming dynasty coinage The early Ming dynasty attempted to use paper currency , with outflows
of bullion limited by its ban on private foreign commerce . Like its forebears , the currency experienced massive counterfeiting and hyperinflation . ( In 1425
, Ming notes were trading at about 0.014 % of its original value under the Hongwu Emperor . ) The notes remained in circulation as late as 1573 but their
printing was ended in 1450 . Minor coins were minted in base metals , but trade mostly occurred using silver ingots . As their purity and exact weight varied ,
they were treated as bullion and measured in tael .
</item 6>

<item 7>
Ming China (1368–1644) had a vast, increasingly commercial economy, supported by agricultural investment, low taxes, cash-crop plantations, rural and national
markets, and expanding wage labor and private manufacturing. Silver gradually replaced failed paper currency, flowing from Japan and Spanish America, but
shortages caused tax crises. Trade with Europe and Japan flourished as state controls weakened and wealthy merchants gained influence. Despite growth and claims
of emerging capitalism, low revenues, fiscal weakness, and unpaid troops contributed
</item 7>

<item 8>
Economy of the Ming dynasty
The entire foreign trade , which was estimated at up to 300 million taels , provided the Ming with a tax of only about 40,000 taels a year . When the Wanli
Emperor sought to increase the salt tax , his measures were opposed by violence and the eunuchs he sent to collect the tax were beheaded by local officials .
Sprouts of capitalism ( edit ) Investment and capital moved off the land and were poured into ventures . Continuing the trend from the Song , Ming investors
poured large amounts of capital into ventures and reaped high profits . Many Chinese scholars believe the Ming was the dynasty in which the `` sprouts of
capitalism '' emerged in China , only to be suppressed by the Qing Dynastry . This theory was widely promoted by Communist scholars during the Maoist period ,
and suffers from that period 's general condemnation of the Manchu Qing , who were accused of mis - managing the Chinese state in the face of foreign
encroachment .
</item 8>

<item 9>
Economy of the Ming dynasty
These privately made `` sycee '' first came into use in Guangdong , spreading to the lower Yangtze sometime before 1423 , the year it became acceptable for
payment of tax obligations . In the mid-15th century , the paucity of circulating silver caused a monetary contraction and extensive reversion to barter . The
problem was met through smuggled , then legal , importation of Japanese silver ( mostly through the Portuguese and Dutch ) and Spanish silver from Potosí
carried on the Manila galleons . Provincial taxes were required to be paid in silver in 1465 ; the salt tax , in 1475 ; and corvée exemptions , in 1485 . By the
late Ming , the amount of silver being used was extraordinary : at a time when English traders considered tens of thousands of pounds an exceptional fortune ,
the Zheng clan of merchants regularly engaged in transactions valued at millions of taels .
</item 9>

<item 10>
Economy of the Ming dynasty
A new class of wage laborers sprung up where none had existed before . In Jingde alone , it was reported that there were no less than 300 pottery factories ,
all operated by wage laborers . Early encouragement of agriculture under Hongwu ( edit ) In order to recover from rule of the Mongols and the wars that followed
them , the Hongwu Emperor enacted pro-agricultural policies . The state invested extensively in agricultural canals , reduced taxes on agriculture to 1 / 30 of
the output , and later to 1.5 % of agricultural output . Ming farmers also introduced many innovations such as water - powered plows , and new agricultural
methods such as crop rotation . This led to a massive agricultural surplus that became the basis of a market economy . Emergence of commercial plantations (
edit ) The Ming saw the rise of commercial plantations who produced crops suitable to their regions . Tea , fruits , paint and other goods were produced on a
massive scale by these agricultural plantations . Regional patterns of production established during this period continued into the Qing dynasty . The Columbian
exchange brought crops such as corn with these foreign crops .
</item 10>
=== END ITEM id=374527481387b3f8 ===

=== ITEM id=bfd09dda0e795c53 ===
Gold sentences and facts:
Sentence 0: The coastline of the island consists of a combination of long beaches , dunes , red sandstone cliffs , salt water marshes and numerous bays and harbours . The beaches , dunes and sandstone cliffs consist of sedimentary rock and other material with a high iron concentration which oxidizes upon exposure to the air . The geological properties of a white silica sand found at Basin Head are unique in the province ; the sand grains cause a scrubbing noise as they rub against each other when walked on . Large dune fields on the north shore can be found on barrier islands at the entrances to various bays and harbours . The magnificent sand dunes at Greenwich are of particular significance . The shifting , parabolic dune system is home to a variety of birds and rare plants and is also a site of significant archeological interest .
  Fact 0.0: The coastline of the island consists of a combination of long beaches, dunes, red sandstone cliffs, salt water marshes, and numerous bays and harbours.
  Fact 0.1: The beaches, dunes, and sandstone cliffs consist of sedimentary rock and other material with a high iron concentration.
  Fact 0.2: That iron concentration oxidizes upon exposure to air.
  Fact 0.3: The geological properties of a white silica sand found at Basin Head are unique in the province.
  Fact 0.4: The sand grains at Basin Head cause a scrubbing noise as they rub against each other when walked on.
  Fact 0.5: Large dune fields on the north shore can be found on barrier islands at the entrances to various bays and harbours.
  Fact 0.6: The magnificent sand dunes at Greenwich are of particular significance.
  Fact 0.7: The shifting, parabolic dune system at Greenwich is home to a variety of birds and rare plants.
  Fact 0.8: The shifting, parabolic dune system at Greenwich is also a site of significant archeological interest.

Retrieved items:
<item 1>
Geography of Prince Edward Island
32,174 ) Demographics Population 139,407 ( 2008 ) Pop . density 24.53 / km ( 63.53 / sq mi ) Ethnic groups Scottish ( 38.0 % ) , English ( 28.7 % ) , Irish (
27.9 % ) , French ( 21.3 % ) , German ( 4.0 % ) , and Dutch ( 3.1 % ) Prince Edward Island 's geography is mostly pastoral with red soil , white sand , and
scattered communities . Known as the `` Garden of the Gulf '' , the island is located in the Gulf of Saint Lawrence north of Nova Scotia and east of New
Brunswick , with which it forms the Northumberland Strait . PEI consists of the capital city Charlottetown , as well as urban towns Cornwall and Stratford and a
developing urban fringe . A smaller urban area surrounds Summerside Harbor , situated on the southern shore 40 kilometers west of Charlottetown Harbor , and
consists primarily of the city of Summerside . As with all natural harbors on the island , Charlottetown and Summerside harbors are created by rias .
</item 1>

<item 2>
Dead Sea
Northwest British Isles English Channel Channel Islands Cotentin Peninsula Normandy Brittany Gulf of Lion Iberia Al - Andalus Baetic System Pyrenees Alpide belt
South Italian Peninsula Insular Italy Tuscan Archipelago Aegadian Islands Iberia Al - Andalus Baetic System Gibraltar Arc Southeastern Mediterranean Crimea
Alpide belt Germanic Celtic Slavic countries Uralic European Plain Eurasian Steppe Pontic -- Caspian steppe Wild Fields Pannonian Basin Great Hungarian Plain
Little Hungarian Plain Eastern Slovak Lowland Regions of North America Northern Eastern Canada Western Canada Canadian Prairies Central Canada Northern Canada
Atlantic Canada The Maritimes French Canada English Canada Acadia Acadian Peninsula Quebec City -- Windsor Corridor Peace River Country Cypress Hills Palliser
's Triangle Canadian Shield Interior Alaska - Yukon lowland taiga Newfoundland ( island ) Vancouver Island Gulf Islands Strait of Georgia Canadian Arctic
Archipelago Labrador Peninsula Gaspé Peninsula Avalon Peninsula Bay de Verde Peninsula Brodeur Peninsula Melville Peninsula Bruce Peninsula Banks Peninsula (
Nunavut ) Cook Peninsula Gulf of Boothia Georgian Bay Hudson Bay James Bay Greenland Pacific Northwest Inland Northwest Northeast New England Mid-Atlantic
Commonwealth West Midwest Upper Midwest Mountain States Intermountain West Basin and Range Province Oregon Trail
</item 2>

<item 3>
Geography of Prince Edward Island
The highest point of land is located at Springton in Queens County ( 152 metres or 499 feet above sea level ) . Contents ( hide ) 1 Communities 2 Coastline 3
See also 4 References 5 External links Communities ( edit ) The smaller rural communities as well as the towns and villages throughout the province proudly
retain a slower - paced , old world flavor , something that factors heavily into Prince Edward Island 's popularity as a destination for relaxation . Most rural
communities on Prince Edward Island are based on small - scale agriculture , given that the size of farm properties is quite small when compared with other
areas in Canada . There is an increasing amount of industrial farming as older farm properties are consolidated and modernized . Coastline ( edit ) The
coastline of the island consists of a combination of long beaches , dunes , red sandstone cliffs , salt water marshes and numerous bays and harbours . The
beaches , dunes and sandstone cliffs consist of sedimentary rock and other material with a high iron concentration which oxidizes upon exposure to the air .
</item 3>

<item 4>
Geography of Prince Edward Island
The geological properties of a white silica sand found at Basin Head are unique in the province ; the sand grains cause a scrubbing noise as they rub against
each other when walked on . Large dune fields on the north shore can be found on barrier islands at the entrances to various bays and harbours . The magnificent
sand dunes at Greenwich are of particular significance . The shifting , parabolic dune system is home to a variety of birds and rare plants and is also a site
of significant archeological interest .
</item 4>

<item 5>
Sub-Saharan Africa
European Plain Eurasian Steppe Pontic -- Caspian steppe Wild Fields Pannonian Basin Great Hungarian Plain Little Hungarian Plain Eastern Slovak Lowland Regions
of North America Northern Eastern Canada Western Canada Canadian Prairies Central Canada Northern Canada Atlantic Canada The Maritimes French Canada English
Canada Acadia Acadian Peninsula Quebec City -- Windsor Corridor Peace River Country Cypress Hills Palliser 's Triangle Canadian Shield Interior Alaska - Yukon
lowland taiga Newfoundland ( island ) Vancouver Island Gulf Islands Strait of Georgia Canadian Arctic Archipelago Labrador Peninsula Gaspé Peninsula Avalon
Peninsula Bay de Verde Peninsula Brodeur Peninsula Melville Peninsula Bruce Peninsula Banks Peninsula ( Nunavut ) Cook Peninsula Gulf of Boothia Georgian Bay
Hudson Bay James Bay Greenland Pacific Northwest Inland Northwest Northeast New England Mid-Atlantic Commonwealth West Midwest Upper Midwest Mountain States
Intermountain West Basin and Range Province Oregon Trail Mormon Corridor Calumet Region Southwest Old Southwest Llano Estacado Central United States Tallgrass
prairie South South Central Deep South Upland South Four Corners East Coast West Coast Gulf Coast Third Coast Coastal states Eastern United States Appalachia
Trans - Mississippi Great North Woods Great Plains Interior Plains Great Lakes Great Basin Great Basin Desert Acadia Ozarks Ark - La - Tex
</item 5>

<item 6>
Ring of Fire
Peninsula Quebec City -- Windsor Corridor Peace River Country Cypress Hills Palliser 's Triangle Canadian Shield Interior Alaska - Yukon lowland taiga
Newfoundland ( island ) Vancouver Island Gulf Islands Strait of Georgia Canadian Arctic Archipelago Labrador Peninsula Gaspé Peninsula Avalon Peninsula Bay de
Verde Peninsula Brodeur Peninsula Melville Peninsula Bruce Peninsula Banks Peninsula ( Nunavut ) Cook Peninsula Gulf of Boothia Georgian Bay Hudson Bay James
Bay Greenland United States Pacific Northwest Inland Northwest Northeast New England Mid-Atlantic Commonwealth West Midwest Upper Midwest Mountain States
Intermountain West Basin and Range Province Oregon Trail Mormon Corridor Calumet Region Southwest Old Southwest Llano Estacado Central United States Tallgrass
prairie South South Central Deep South Upland South Four Corners East Coast West Coast Gulf Coast Third Coast Coastal states Eastern United States Appalachia
Trans - Mississippi Great North Woods Great Plains Interior Plains Great Lakes Great Basin Great Basin Desert Acadia Ozarks Ark - La - Tex Waxhaws Siouxland
Twin Tiers Driftless Area Palouse Piedmont Atlantic coastal plain Outer Lands Black Dirt Region Blackstone Valley Piney Woods Rocky Mountains Mojave Desert The
Dakotas The Carolinas Shawnee Hills San Fernando Valley Tornado Alley North Coast Lost Coast Emerald Triangle San Francisco Bay
</item 6>

<item 7>
Geography of Prince Edward Island
Geography of Prince Edward Island - wikipedia Geography of Prince Edward Island Jump to : navigation , search This article contains wording that promotes the
subject in a subjective manner without imparting real information . Please remove or replace such wording and instead of making proclamations about a subject 's
importance , use facts and attribution to demonstrate that importance . ( August 2017 ) ( Learn how and when to remove this template message ) Prince Edward
Island The coast of Prince Edward Island around Cavendish Geography Location Gulf of Saint Lawrence Area 5,683.91 km ( 2,194.57 sq mi ) Highest elevation 152 m
( 499 ft ) Administration Canada Province Prince Edward Island Largest settlement Charlottetown ( pop . 32,174 ) Demographics Population 139,407 ( 2008 ) Pop .
</item 7>

<item 8>
Geography of Prince Edward Island
( See also a list of communities in Prince Edward Island . ) The island 's landscape is pastoral : rolling hills , pristine forests , white sand beaches , ocean
coves and the famous red soil have given Prince Edward Island a reputation as a province of outstanding natural beauty . A number of laws have been passed by
the provincial government to attempt to preserve the landscape through regulation , although the lack of consistent enforcement and absence of province - wide
zoning and land - use planning has resulted in some aesthetically displeasing development in recent years . The island 's lush landscape has had a strong
bearing not only on its economy but also its culture . Author Lucy Maud Montgomery drew inspiration from the land during the late Victorian Era for the setting
of her classic novel Anne of Green Gables . Today , many of the same qualities that Montgomery and others found in the Island are enjoyed by tourists who visit
during all seasons . They enjoy a variety of leisure activities , including beaches , various golf courses , eco-tourism adventures , and simply touring the
countryside and enjoying cultural events in local communities around the island .
</item 8>

<item 9>
Geography of Prince Edward Island
See also ( edit ) Geography of Canada portal List of parks in Prince Edward Island List of islands of Prince Edward Island References ( edit ) External links (
edit ) Geography of the provinces and territories of Canada National Geography of Canada Centre of Canada Provinces British Columbia Alberta Saskatchewan
Manitoba Ontario Quebec New Brunswick Nova Scotia Prince Edward Island Newfoundland and Labrador Territories Yukon Northwest Territories Nunavut See also
Regions of Canada Islands Rivers Lakes Mountains National Parks Volcanoes Portal WikiProject Retrieved from ``
https://en.wikipedia.org/w/index.php?title=Geography_of_Prince_Edward_Island&oldid=835998214 '' Categories : Geography of Prince Edward Island Hidden categories
: Articles with peacock terms from August 2017 All articles with peacock terms Talk Contents About Wikipedia Français Edit links This page was last edited on 12
April 2018 , at 01 : 54 . Text is available under the Creative Commons Attribution - ShareAlike License ; additional terms may apply . By using this site , you
agree to the Terms of Use and Privacy Policy . Wikipedia ® is a registered trademark of the Wikimedia Foundation , Inc. , a non-profit organization . About
Wikipedia
</item 9>

<item 10>
Prince Edward Island, Canada, lies in the Gulf of St. Lawrence between Nova Scotia and New Brunswick and covers 5,684 km². Its highest point, near Springton, is
152 m. The largely pastoral landscape features rolling hills, forests, red iron-rich soil, beaches, dunes, sandstone cliffs, marshes, bays, and harbours.
Charlottetown and Summerside are major urban centres; rural communities have traditionally relied on small-scale farming. Notable
</item 10>
=== END ITEM id=bfd09dda0e795c53 ===

=== ITEM id=fcab0e7972b07830 ===
Gold sentences and facts:
Sentence 0: `` The Gift of the Magi '' is a short story , written by O. Henry ( a pen name for William Sydney Porter ) , about a young husband and wife and how they deal with the challenge of buying secret Christmas gifts for each other with very little money . As a sentimental story with a moral lesson about gift - giving , it has been a popular one for adaptation , especially for presentation at Christmas time . The plot and its twist ending are well - known , and the ending is generally considered an example of comic irony . It was allegedly written at Pete 's Tavern on Irving Place in New York City .
  Fact 0.0: The Gift of the Magi is a short story.
  Fact 0.1: The Gift of the Magi was written by O. Henry.
  Fact 0.2: O. Henry is a pen name for William Sydney Porter.
  Fact 0.3: The Gift of the Magi is about a young husband and wife.
  Fact 0.4: The husband and wife deal with the challenge of buying secret Christmas gifts for each other with very little money.
  Fact 0.5: The Gift of the Magi is a sentimental story with a moral lesson about gift-giving.
  Fact 0.6: The Gift of the Magi has been a popular story for adaptation.
  Fact 0.7: The Gift of the Magi has been especially popular for presentation at Christmas time.
  Fact 0.8: The plot of The Gift of the Magi is well-known.
  Fact 0.9: The twist ending of The Gift of the Magi is well-known.
  Fact 0.10: The ending of The Gift of the Magi is generally considered an example of comic irony.
  Fact 0.11: The Gift of the Magi was allegedly written at Pete's Tavern on Irving Place in New York City.

Retrieved items:
<item 1>
The Gift of the Magi
The special premiered on HBO on December 17 , 1978 . The special later aired on ABC in 1980 and on Nickelodeon in the 1990s . The special features several
original songs written by songwriter Paul Williams . The sketch comedy show Studio C contains a skit called `` Gift of the Magi '' in which Jim and Della argue
over their gift - giving . The television series My Little Pony : Friendship is Magic has the episode `` The Gift of the Maud Pie '' in which Pinkie Pie gives
Maud a perfect gift , costing her the confetti - using Party Cannon . Maud 's gift in return was confetti for the cannon of which Pinkie Pie had sold . Radio
drama series Adventures in Odyssey features a comedic adaptation of this story in its episode `` Gifts for Madge & Guy '' The animated sketch comedy series
Robot Chicken features a parody of the story in its fourth season , except that Jim does not sell his watch and instead buys Della lingerie , much to her
consternation . Joni Mitchell wrote and performed , but never recorded , a song based on the story .
</item 1>

<item 2>
The Gift of the Magi
Unusually late , Jim walks in and immediately stops short at the sight of Della , who had previously prayed that she was still pretty to Jim . Della then admits
to Jim that she sold her hair to buy him his present . Jim gives Della her present -- an assortment of combs , useless now that her hair is short . Della then
shows Jim the chain she bought for him , to which Jim says he sold his watch to get the money to buy her combs . Although Jim and Della are now left with gifts
that neither one can use , they realize how far they are willing to go to show their love for each other , and how priceless their love really is . The story
ends with the narrator comparing the pair 's mutually sacrificial gifts of love with those of the Biblical Magi .
</item 2>

<item 3>
The Gift of the Magi
Text is available under the Creative Commons Attribution - ShareAlike License ; additional terms may apply . By using this site , you agree to the Terms of Use
and Privacy Policy . Wikipedia ® is a registered trademark of the Wikimedia Foundation , Inc. , a non-profit organization . About Wikipedia
</item 3>

<item 4>
The Gift of the Magi
Henry The Gift of the Magi public domain audiobook at LibriVox O'Henry's `` The Gift of the Magi '' : Common Core Exemplar from NEH 's EDSITEment `` The Gift of
the Magi '' study guide , teacher resources , themes , quotes ( hide ) O. Henry Short stories `` The Duplicity of Hargraves '' ( 1902 ) `` A Retrieved
Reformation '' ( 1903 ) `` The Cop and the Anthem '' ( 1904 ) `` The Gift of the Magi '' ( 1905 ) `` Conscience in Art '' ( 1907 ) `` The Last Leaf '' ( 1907 )
`` The Third Ingredient '' ( 1908 ) `` The Ransom of Red Chief '' ( 1910 ) `` Makes the Whole World Kin '' ( 1911 ) Novels Cabbages and Kings ( 1904 ) Story
collections The Four Million ( 1906 ) Roads of Destiny ( 1909 ) Whirligigs ( 1910 ) Waifs and Strays ( 1917 ) Related O. Henry Award William Sidney Porter House
O. Henry House and Museum O. Henry Hall O.
</item 4>

<item 5>
The Gift of the Magi
The Gift of the Magi - wikipedia The Gift of the Magi Jump to : navigation , search This article is about the short story . For the Magi that visited baby Jesus
, see Biblical Magi . For other uses , see The Gift of the Magi ( disambiguation ) . `` The Gift of the Magi '' Author O. Henry Country United states Language
English Genre ( s ) Short story Published in The Four Million Publication type Anthology Publication date December 10 , 1905 ( newspaper ) ; April 10 , 1906 (
book ) `` The Gift of the Magi '' is a short story , written by O. Henry ( a pen name for William Sydney Porter ) , about a young husband and wife and how they
deal with the challenge of buying secret Christmas gifts for each other with very little money . As a sentimental story with a moral lesson about gift - giving
, it has been a popular one for adaptation , especially for presentation at Christmas time . The plot and its twist ending are well - known , and the ending is
generally considered an example of comic irony . It was allegedly written at Pete 's Tavern on Irving Place in New York City .
</item 5>

<item 6>
The Gift of the Magi
Adaptations ( edit ) The story has been adapted to films , The Sacrifice ( 1909 ) , Love 's Surprises Are Futile ( 1916 ) , The Gift of the Magi ( 1917 ) , a
segment of O. Henry 's Full House ( 1952 ) , The Gift of Love ( 1978 ) , The Gift of the Magi ( 1958 ) , Dary magów ( Poland , 1972 ) , Christmas Eve on Sesame
Street ( 1978 ) , I 'll not be a gangster , love ( Не буду гангстером , дорогая / Nebūsiu gangsteriu , brangioji , USSR , 1978 ) , Mickey 's Once Upon a
Christmas ( 1999 ) , The Gift of the Magi ( 2004 ) and the short film for the Irish band The Script in 2010 called For the First Time . Love , another French
movie , based some of its scenes on this story . Raincoat ( 2004 ) , a Hindi film directed by Rituparno Ghosh is an adaptation of the story .
</item 6>

<item 7>
The Gift of the Magi
The Greek film directed by Ismene Daskarolis ( 2014 ) places it in the economical crisis of Greece today . The Mexican film Nosotros los pobres includes this
tale as a small sub-plot . There is also a Bulgarian short film adaptation known as `` Darovete na vlahvite '' ( 2013 ) directed by Ivan Abadjiev . An off -
Broadway musical version entitled The Gifts of the Magi premiered at Lamb 's Theatre in New York City in 1984 . Written by Mark St. Germain and Randy Courts ,
the play is regularly produced in schools and regional theaters . In 1955 , in The Honeymooners episode 13 of season 1 , `` ' Twas the Night Before Christmas ''
involves Ralph looking for the perfect present for Alice for Christmas . Without money , he pawns his prized bowling ball to buy her an expensive gift , only to
find out in the end that she bought him a custom bowling ball bag .
</item 7>

<item 8>
The Gift of the Magi
LiteraryTraveler.com . Jump up ^ Не буду гангстером , дорогая Jump up ^ Mickey 's Once Upon A Christmas on IMDb Jump up ^ O. Henry on IMDb Jump up ^ Raincoat on
IMDb Jump up ^ The Gift of the Magi ( 2014 ) - IMDb Jump up ^ Darovete Na Vlahvite ( 2013 ) - IMDb Jump up ^ St. Germain , Mark ; Courts , Randy ( 1984 ) . The
Gifts of the Magi . New York : Dramatists Play Service , Inc . ISBN 978 - 0 - 8222 - 1461 - 8 Jump up ^ Watch A Trump Christmas from Saturday Night Live on
NBC.com , retrieved 2017 - 08 - 03 External links ( edit ) Wikisource has original text related to this article : The Gift of the Magi `` The Gift of the Magi
'' by O.
</item 8>

<item 9>
The Gift of the Magi
Henry Award William Sidney Porter House O. Henry House and Museum O. Henry Hall O. Henry 's Full House ( 1952 ) Strictly Business ( 1962 ) The Trust That Went
Bust Retrieved from `` https://en.wikipedia.org/w/index.php?title=The_Gift_of_the_Magi&oldid=816945958 '' Categories : 1906 short stories Christmas short
stories Irony Short stories by O. Henry Short stories adapted into films Works originally published in the New York World Hidden categories : Pages using
deprecated image syntax Articles with LibriVox links Talk Contents About Wikipedia Wikisource Deutsch فارسی 한국어 Հայերեն 日本 語 ਪੰਜਾਬੀ Português Русский Simple
English Slovenščina Tagalog Українська Tiếng Việt 中文 Edit links This page was last edited on 24 December 2017 , at 22 : 11 . Text is available under the
Creative Commons Attribution - ShareAlike License ; additional terms may apply .
</item 9>

<item 10>
The Gift of the Magi
It was allegedly written at Pete 's Tavern on Irving Place in New York City . The story was initially published in The New York Sunday World under the title ``
Gifts of the Magi '' on December 10 , 1905 . It was first published in book form in the O. Henry Anthology The Four Million in April 1906 . Contents ( hide ) 1
Summary 2 Adaptations 3 References 4 External links Summary ( edit ) Mr. James Dillingham Young ( `` Jim '' ) and his wife , Della , are a couple living in a
modest apartment . On Christmas Eve , with only $1.87 in hand , and desperate to find a gift for Jim , Della sells her hair for $20 to a nearby hairdresser
named Madame Sofronie , and eventually finds a platinum pocket watch fob chain for Jim 's watch for $21 . Satisfied with the perfect gift for Jim , Della runs
home and begins to prepare pork chops for dinner . At 7 o'clock , Della sits at a table near the door , waiting for Jim to come home .
</item 10>
=== END ITEM id=fcab0e7972b07830 ===

=== ITEM id=50ca6ffd28632436 ===
Gold sentences and facts:
Sentence 0: Year Film Role Notes Dumb and Dumber State Trooper Down Periscope E.T. `` Sonar '' Lovacelli RocketMan Fred Z . Randall Wag the Dog Pet Wrangler Space Ghost Coast to Coast Himself 1 episode 1998 Half Baked Kenny Davis There 's Something About Mary Hitchhiker ( uncredited ) Dog Park Callum Ned 's Newt Newton ( voice ) Mr. Headmistress Tucker 1999 Superstar Slater 2000 The Whole Nine Yards Special Agent Steve Hanson Big Money Hustlas Officer Harry Cox Becoming Dick Richard Breggs Freddy Got Fingered Darren Gary & Mike Mike Bonner ( voice ) 2002 Back by Midnight Sheriff Hubbard Sorority Boys Doofer / Roberta 2003 Family Tree Jake Kart Racer Zee 2005 Lucky 13 Bleckman Because of Winn - Dixie Policeman Robots Lugnut ( voice ) 2006 Employee of the Month Russell Porpis - Gunders Surf School Rip Ergo Proxy Head Dog Cop ( voice ) 1 episode 2007 Meet the Robinsons Carl ( voice ) The Dukes of Hazzard : The Beginning Sheriff Rosco P. Coltrane 2008 Bachelor Party 2 : The Last Temptation Derek Madagascar : Escape 2 Africa Giraffe ( voice ) 2009 Fudgy Wudgy Fudge Face Elmore P. Fudge My Life in Ruins Big Al Sawchuck 2009 The Haunted World of El Superbeasto Gerard the Exterminator ( voice ) 2011 Spooky Buddies Warwick the Warlock Lloyd the Conqueror Vulcan Dahmer Vs. Gacy God Dan Vs . Hiram Episode : Technology Robot Chicken Professor Cornelius Q. Quibblefingers ( voice ) 1 episode ; also writes 4 episodes 2012 The High Fructose Adventures of Annoying Orange Apple ( voice ) Robot and Monster Monster ( voice ) Main Role 2014 Back in the Day Skunk 2015 Jake and the Never Land Pirates Captain Frost ( voice ) Penn Zero : Part - Time Hero Helper Hue ( voice ) 2016 Sausage Party Ketchup Bottle / Drug Dealer / Baba Ganoush ( voice ) Skylanders Academy Hugo ( voice ) 2017 - present Puppy Dog Pals Bob ( voice ) Also show creator
  Fact 0.0: The listed performer had the role State Trooper in Dumb and Dumber.
  Fact 0.1: The listed performer appeared in Down Periscope, with no role given.
  Fact 0.2: The listed performer had the role 'Sonar' Lovacelli in E.T.
  Fact 0.3: The listed performer had the role Fred Z. Randall in RocketMan.
  Fact 0.4: The listed performer had the role Pet Wrangler in Wag the Dog.
  Fact 0.5: The listed performer had the role Himself in Space Ghost Coast to Coast, for 1 episode.
  Fact 0.6: In 1998, the listed performer had the role Kenny Davis in Half Baked.
  Fact 0.7: In 1998, the listed performer had the role Hitchhiker, uncredited, in There's Something About Mary.
  Fact 0.8: In 1998, the listed performer had the role Callum in Dog Park.
  Fact 0.9: In 1998, the listed performer had the role Newton, a voice role, in Ned's Newt.
  Fact 0.10: In 1998, the listed performer had the role Tucker in Mr. Headmistress.
  Fact 0.11: In 1999, the listed performer had the role Slater in Superstar.
  Fact 0.12: In 2000, the listed performer had the role Special Agent Steve Hanson in The Whole Nine Yards.
  Fact 0.13: In 2000, the listed performer had the role Officer Harry Cox in Big Money Hustlas.
  Fact 0.14: In 2000, the listed performer had the role Richard Breggs in Becoming Dick.
  Fact 0.15: In 2000, the listed performer had the role Darren in Freddy Got Fingered.
  Fact 0.16: In 2000, the listed performer had the role Mike Bonner, a voice role, in Gary & Mike.
  Fact 0.17: In 2002, the listed performer had the role Sheriff Hubbard in Back by Midnight.
  Fact 0.18: In 2002, the listed performer had the role Doofer / Roberta in Sorority Boys.
  Fact 0.19: In 2003, the listed performer had the role Jake in Family Tree.
  Fact 0.20: In 2003, the listed performer had the role Zee in Kart Racer.
  Fact 0.21: In 2005, the listed performer had the role Bleckman in Lucky 13.
  Fact 0.22: In 2005, the listed performer had the role Policeman in Because of Winn-Dixie.
  Fact 0.23: In 2005, the listed performer had the role Lugnut, a voice role, in Robots.
  Fact 0.24: In 2006, the listed performer had the role Russell Porpis-Gunders in Employee of the Month.
  Fact 0.25: In 2006, the listed performer had the role Rip in Surf School.
  Fact 0.26: In 2006, the listed performer had the role Head Dog Cop, a voice role, for 1 episode, in Ergo Proxy.
  Fact 0.27: In 2007, the listed performer had the role Carl, a voice role, in Meet the Robinsons.
  Fact 0.28: In 2007, the listed performer had the role Sheriff Rosco P. Coltrane in The Dukes of Hazzard: The Beginning.
  Fact 0.29: In 2008, the listed performer had the role Derek in Bachelor Party 2: The Last Temptation.
  Fact 0.30: In 2008, the listed performer had the role Giraffe, a voice role, in Madagascar: Escape 2 Africa.
  Fact 0.31: In 2009, the listed performer had the role Fudge Face Elmore P. Fudge in Fudgy Wudgy.
  Fact 0.32: In 2009, the listed performer had the role Big Al Sawchuck in My Life in Ruins.
  Fact 0.33: In 2009, the listed performer had the role Gerard the Exterminator, a voice role, in The Haunted World of El Superbeasto.
  Fact 0.34: In 2011, the listed performer had the role Warwick the Warlock, Lloyd the Conqueror, and Vulcan in Spooky Buddies.
  Fact 0.35: In 2011, the listed performer had the role God in Dahmer Vs. Gacy.
  Fact 0.36: In 2011, the listed performer had the role Robot in Dan Vs. Hiram, Episode: Technology.
  Fact 0.37: In 2011, the listed performer had the role Professor Cornelius Q. Quibblefingers, a voice role, for 1 episode, in Robot Chicken.
  Fact 0.38: The listed performer also wrote 4 episodes of Robot Chicken.
  Fact 0.39: In 2012, the listed performer had the role Apple, a voice role, in The High Fructose Adventures of Annoying Orange.
  Fact 0.40: In 2012, the listed performer had the role Monster, a voice role and a main role, in Robot and Monster.
  Fact 0.41: In 2014, the listed performer had the role Skunk in Back in the Day.
  Fact 0.42: In 2015, the listed performer had the role Captain Frost, a voice role, in Jake and the Never Land Pirates.
  Fact 0.43: In 2015, the listed performer had the role Helper Hue, a voice role, in Penn Zero: Part-Time Hero.
  Fact 0.44: In 2016, the listed performer had the role Ketchup Bottle / Drug Dealer / Baba Ganoush, a voice role, in Sausage Party.
  Fact 0.45: In 2016, the listed performer had the role Hugo, a voice role, in Skylanders Academy.
  Fact 0.46: From 2017 to the present, the listed performer had the role Bob, a voice role, in Puppy Dog Pals.
  Fact 0.47: The listed performer is also the creator of Puppy Dog Pals.

Retrieved items:
<item 1>
Graph query row:
person: Harland Williams
relationship: RELATED_TO
work: Dumb and Dumber
role: State Trooper
fact: None
relationship_detail: None
relation_detail: None
</item 1>

<item 2>
Harland Williams
Headmistress Tucker 1999 Superstar Slater 2000 The Whole Nine Yards Special Agent Steve Hanson Big Money Hustlas Officer Harry Cox Becoming Dick Richard Breggs
Freddy Got Fingered Darren Gary & Mike Mike Bonner ( voice ) 2002 Back by Midnight Sheriff Hubbard Sorority Boys Doofer / Roberta 2003 Family Tree Jake Kart
Racer Zee 2005 Lucky 13 Bleckman Because of Winn - Dixie Policeman Robots Lugnut ( voice ) 2006 Employee of the Month Russell Porpis - Gunders Surf School Rip
Ergo Proxy Head Dog Cop ( voice ) 1 episode 2007 Meet the Robinsons Carl ( voice ) The Dukes of Hazzard : The Beginning Sheriff Rosco P. Coltrane 2008 Bachelor
Party 2 : The Last Temptation Derek Madagascar : Escape 2 Africa Giraffe ( voice ) 2009 Fudgy Wudgy Fudge Face Elmore P.
</item 2>

<item 3>
California Highway Patrol
Peyer declined to provide a sample , nor explain why he declined to do so . Media references ( edit ) CHiPs was a fictional television drama show ( 1977 - 1983
) about the CHP , and the CHP helped out in the 1955 TV show Highway Patrol starring Broderick Crawford . Also the CHiPs TV series was made into a modern
television film called CHiPs ' 99 . A comical motion picture version of the show has been released in 2017 ( CHiPS ) but it has not been endorsed by the CHP at
all . Former CHP Commissioner Maury Hannigan was the host of the syndicated TV series Real Stories of the Highway Patrol , which aired in the mid-to - late
1990s . The show featured not just the CHP , but also the Minnesota State Patrol , the Nevada Highway Patrol , the Alaska State Troopers , the South Dakota
Highway Patrol , the Rhode Island State Police , the Massachusetts State Police , the West Virginia State Police and many other state police and highway patrol
agencies in the United State of America , and in one episode , the Royal Newfoundland Constabulary .
</item 3>

<item 4>
The Dukes of Hazzard (film)
The characters are now campus security officers , who give a warning to the Duke boys for driving too slowly . Paul Soter appears as TV newsman Rick Shakely ,
who is reporting from the Hazzard Road Rally . Charlie Finn , who played a dimwitted fast - food employee in Super Troopers , appears as a dimwitted geology
student who assists the Duke boys with the coal samples . Other notable cameos During the bar - fight scene , Indy - car driver A.J. Foyt IV appears as himself
. Bloopers roll under the ending credits , one of which features Rip Taylor interrupting the bedroom scene with Luke and the two college girls . Taylor had
previously appeared with Knoxville in Jackass : The Movie . Production ( edit ) Stunts ( edit ) Knoxville said he was initially reluctant to take on the role ,
but was persuaded by script changes and the presence of Dan Bradley as stunt coordinator and second unit for the car chase scenes . Knoxville praised him ,
saying , `` everyone in Hollywood wants Dan Bradley to shoot their car stuff '' . Locations ( edit ) The majority of the film was shot in and around Clinton ,
Louisiana .
</item 4>

<item 5>
The Dukes of Hazzard (film)
Gainey as Sheriff Rosco P. Coltrane Michael Weston as Deputy Enos Strate Lynda Carter as Pauline James Roday as Billy Prickett Kevin Heffernan as Derek `` Sheev
'' Sheevington Nikki Griffin as Katie - Lynn Johnson Jacqui Maxwell as Annette Alice Greczyn as Laurie Pullman Junior Brown as The Balladeer ( narrator ) Joe
Don Baker as Governor Jim Applewhite Barry Corbin as Bill Pullman Andrew Prine as Angry Man Randy Orton as Bo Duke ( Stunt double ) Cameos ( edit ) All five
members of the comedy film troupe Broken Lizard make appearances in the film , classified as cameos , except for Kevin Heffernan , who had a larger speaking
role ( Sheev ) . Broken Lizard cameos Steve Lemme appears as Jimmy Pullman , the son of Bill Pullman , in an early car chase scene in which he accidentally
shoots the inside of his father 's truck . Jay Chandrasekhar and Erik Stolhanske reprise their roles as Ramathorn and Rabbit from the Broken Lizard comedy ,
Super Troopers .
</item 5>

<item 6>
Harland Williams
Williams created the Disney Junior series Puppy Dog Pals that debuted in 2017 . Filmography ( edit ) Year Film Role Notes Dumb and Dumber State Trooper Down
Periscope E.T. `` Sonar '' Lovacelli RocketMan Fred Z . Randall Wag the Dog Pet Wrangler Space Ghost Coast to Coast Himself 1 episode 1998 Half Baked Kenny
Davis There 's Something About Mary Hitchhiker ( uncredited ) Dog Park Callum Ned 's Newt Newton ( voice ) Mr.
</item 6>

<item 7>
The Dukes of Hazzard (film)
The cousins ' primary mode of transportation is an orange 1969 Dodge Charger that the boys affectionately refer to as the `` General Lee '' . Along the way ,
the family is tormented by corrupt Hazzard County Commissioner Jefferson Davis Hogg , widely known as `` Boss Hogg '' ( Burt Reynolds ) , and his willing but
dimwitted henchman , Sheriff Rosco P. Coltrane ( M.C. Gainey ) . After Rosco has the General Lee impounded after Bo and Luke 's attempt to run away from a
daughter of one their many moonshine customers , Billy Prickett ( James Roday ) , a famous stock - car driver , enters Hazzard to participate in the rally .
Meanwhile , Rosco plants a fake moonshine still ( `` ' cause he 's too dumb to find the real one '' ) in Uncle Jesse 's barn and seizes the Duke property in the
interest of eminent domain for Boss Hogg , forcing the family to temporarily reside with neighbor and Uncle Jesse 's love - interest , Pauline ( Lynda Carter )
.
</item 7>

<item 8>
California Highway Patrol
They spotted the car , fell in behind , called for backup , and began the stop procedure . When the suspects ' vehicle came to a halt in the parking lot , the
driver was instructed to step out of the vehicle and spread his hands on the hood . Gore approached him and Frago moved to the passenger side . The passenger
side door suddenly swung open and the passenger sprung out , firing at Frago , who fell with two shots in his chest . The gunman , later identified as Jack
Twinning , then turned and fired once at Gore , who returned fire . In that moment the driver , Bobby Davis , turned and shot Gore twice at close range . Both
officers died instantly . When Officers James Pence and George Alleyn drove in moments later , they could not see suspects or other officers , but both
immediately came under fire . Pence put out an 11 ‐ 99 call ( `` officer down '' ) then took cover behind the passenger door . Alleyn grabbed his shotgun , and
positioned himself behind the driver - side door . Both officers were mortally wounded in the ensuing exchange , and one suspect was hit .
</item 8>

<item 9>
Geoffrey Palmer (actor)
( 1973 ) as Examinator Doctor / Basil Keyes The Battle of Billy 's Pond ( 1976 ) - First Policeman The Outsider ( 1979 ) as Colonel Wyndham The Honorary Consul
( 1983 ) as Belfrage : British Ambassador A Zed & Two Noughts ( 1985 ) as Fallast Clockwise ( 1986 ) as Headmaster A Fish Called Wanda ( 1988 ) as Judge Hawks (
1988 ) as SAAB Salesman The Madness of King George ( 1994 ) as Warren Mrs. Brown ( 1997 ) as Henry Ponsonby Tomorrow Never Dies ( 1997 ) as Admiral Roebuck
Stiff Upper Lips ( 1998 ) as His Butler 's Voice Anna and the King ( 1999 ) as Lord John Bradley Rat ( 2000 ) as The Doctor Peter Pan ( 2003 ) as Sir Edward
Quiller Couch Picadilly Jim ( 2005 ) - Bayliss The Pink Panther 2 ( 2009 ) as Joubert W.E. ( 2011 ) as Stanley Baldwin Lost Christmas ( 2011 ) as Dr.
</item 9>

<item 10>
The Dukes of Hazzard (film)
They are later arrested by Atlanta Police after running from campus police . Back in Hazzard , Daisy learns , with the help of Sheriff 's Deputy Enos Strate (
Michael Weston ) , that Billy Prickett has been hired by Boss Hogg to participate in the rally as a ringer . Boss Hogg then heads to Atlanta , where he informs
the Duke boys , in lock - up , that they are too late to stop him and reveals that the vote on Hogg 's proposition is at the same time as the rally , explaining
Billy Prickett 's involvement . During a transfer from detainment , Daisy helps the boys escape from the patrol car , and they speed home to try to inform the
townsfolk , escaping Atlanta Police , and the Georgia State Patrol after Bo outmaneuvers the city cops . Upon returning home , the Dukes discover that Boss Hogg
and Rosco had taken Uncle Jesse and Pauline hostage , an obvious trap for the boys , and that Billy is in on the scheme because he is ashamed of the town 's low
status .
</item 10>
=== END ITEM id=50ca6ffd28632436 ===
