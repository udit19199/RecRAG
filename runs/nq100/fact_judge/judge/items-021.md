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

This file has 5 items.

=== ITEM id=fc0259790665c268 ===
Gold sentences and facts:
Sentence 0: Spike , occasionally referred to as Butch or Killer , is a stern but occasionally dumb American bulldog who is particularly disapproving of cats , but a softie when it comes to mice ( though in his debut appearance , Dog Trouble , Spike goes after both Tom and Jerry ) , and later , his son Tyke . In the shorts Jerry would often try to get Tom in trouble with Spike making him a shoo - in for a beating from the bulldog . Spike has a few weaknesses that Tom tries to capitalize upon : his possessiveness about his bone and his ticklishness . He made his first appearance in the 1942 Tom and Jerry cartoon Dog Trouble , and his first speaking role was in 1944 's The Bodyguard , where he was voiced by Billy Bletcher up until 1949 , from which point he was voiced by Daws Butler . Unlike his father Spike , Tyke does not speak . He only communicates by yapping , whimpering , growling , facial expressions and wagging his tail . In Tom and Jerry Kids , Tyke does have a speaking role in the program and is the first time that viewers were able to hear Tyke speak . Spike is very protective towards his son and gets very angry at Tom if Tyke is bothered or harmed . Although Tyke has spoken in Tom and Jerry Kids , he has laughed in one Tom and Jerry short . After Daws Butler , Maurice LaMarche , Frank Welker , John DiMaggio , Michael Donovan , Phil LaMarr and currently Rick Zieff would all perform Spike 's voice . Tyke 's vocal effects are provided by Frank Welker and speaking roles by Patric Zimmerman .
  Fact 0.0: Spike is occasionally referred to as Butch or Killer.
  Fact 0.1: Spike is a stern but occasionally dumb American bulldog.
  Fact 0.2: Spike is particularly disapproving of cats.
  Fact 0.3: Spike is a softie when it comes to mice.
  Fact 0.4: In his debut appearance, Dog Trouble, Spike goes after both Tom and Jerry.
  Fact 0.5: Spike has a son named Tyke.
  Fact 0.6: In the shorts, Jerry would often try to get Tom in trouble with Spike.
  Fact 0.7: Spike makes Tom a shoo-in for a beating from the bulldog.
  Fact 0.8: Spike has a few weaknesses that Tom tries to capitalize upon.
  Fact 0.9: Spike's first weakness is his possessiveness about his bone.
  Fact 0.10: Spike's second weakness is his ticklishness.
  Fact 0.11: Spike made his first appearance in the 1942 Tom and Jerry cartoon Dog Trouble.
  Fact 0.12: Spike's first speaking role was in 1944's The Bodyguard.
  Fact 0.13: Billy Bletcher voiced Spike in The Bodyguard and continued until 1949.
  Fact 0.14: From 1949, Daws Butler voiced Spike.
  Fact 0.15: Tyke, unlike his father Spike, does not speak.
  Fact 0.16: Tyke only communicates by yapping, whimpering, growling, facial expressions, and wagging his tail.
  Fact 0.17: In Tom and Jerry Kids, Tyke has a speaking role.
  Fact 0.18: Tom and Jerry Kids is the first time viewers were able to hear Tyke speak.
  Fact 0.19: Spike is very protective towards his son Tyke.
  Fact 0.20: Spike gets very angry at Tom if Tyke is bothered or harmed.
  Fact 0.21: Although Tyke has spoken in Tom and Jerry Kids, Tyke has laughed in one Tom and Jerry short.
  Fact 0.22: Maurice LaMarche performed Spike's voice after Daws Butler.
  Fact 0.23: Frank Welker performed Spike's voice after Daws Butler.
  Fact 0.24: John DiMaggio performed Spike's voice after Daws Butler.
  Fact 0.25: Michael Donovan performed Spike's voice after Daws Butler.
  Fact 0.26: Phil LaMarr performed Spike's voice after Daws Butler.
  Fact 0.27: Rick Zieff currently performs Spike's voice.
  Fact 0.28: Frank Welker provides Tyke's vocal effects.
  Fact 0.29: Patric Zimmerman provides Tyke's speaking roles.

Retrieved items:
<item 1>
List of Tom and Jerry characters
Like many other early human characters , her face was never shown . Chuck Jones era ( edit ) The following characters were introduced in the theatrical shorts
that were produced by Chuck Jones in the 1960s . Tiny bulldog ( edit ) This nameless mouse - sized bulldog , designed similarly to Spike the Bulldog , is Jerry
's pet dog companion in only two shorts : The Cat 's Me - Ouch and Purr - Chance to Dream . His role is similar to Spike in cartoons such as The Bodyguard , Fit
To Be Tied , and Much Ado About Mousing , which is to defend Jerry against Tom . He is presumably the oldest son of Spike before Tyke , and this could be
identified by his looks and personality that is similar to his father Spike . Unlike Spike and Tyke he has a big bite ( similar to those of Looney Tunes '
Tasmanian Devil ) . Porpoise ( edit ) A shark who tries to eat Tom and Jerry . He appears in Cannery Rodent , Puss ' n ' Boats , Filet Meow and Surf - Bored Cat
.
</item 1>

<item 2>
List of Tom and Jerry characters
He is based on the character from Robin Hood tales . Yvonne jockalong ( edit ) Love interest to both Tom and Jerry , she appears in Sky Bunny and The Tennis
Menace . Sergeant farce ( edit ) A police officer who appears in See Dr. Jackal and Hide and in The Police Kitten where he is the superior officer to both Tom
and Jerry . Alley Cat Gang ( edit ) The Tom and Jerry Comedy Show ( edit ) The following characters were introduced in the television series The Tom and Jerry
Comedy Show . Droopy ( edit ) Main article : Droopy Dog A Basset Hound often with a miserable look on his face , originally introduced in the Tex Avery shorts
of the 1940s . He first made a poster cameo in Matinee Mouse . He also appeared in the Droopy and Dripple segments of Tom and Jerry Kids . He is sometimes an
adversary of Spike . He would also appear in Tom and Jerry Tales and would appear in the direct - to - video films .
</item 2>

<item 3>
List of Tom and Jerry characters
References ( edit ) Jump up ^ Deitch , Gene ( 2015 ) . Tom and Jerry ... and Gene in Tom and Jerry : The Gene Deitch Collection ( DVD ) . 20th Century Fox . Tom
and Jerry Characters Tom Cat Jerry Mouse Butch Nibbles Spike and Tyke Mammy Two Shoes Theatrical releases Tom and Jerry ( shorts ) Spike and Tyke ( shorts ) The
Movie Cameos Anchors Aweigh Dangerous When Wet DVD releases of shorts Spotlight Collection Golden Collection Chuck Jones Collection Classic Collection Deluxe
Anniversary Collection Gene Deitch Collection Television series Show ( 1975 ) Comedy Show Kids Show episodes Tales Show ( 2014 ) episodes Television specials
Hanna - Barbera 's 50th : A Yabba Dabba Doo Celebration The Mansion Cat Santa 's Little Helpers Direct - to - video films The Magic Ring Blast Off to Mars The
Fast and the Furry Shiver Me Whiskers A Nutcracker Tale Meet Sherlock Holmes The Wizard of Oz Robin Hood and His Merry Mouse Giant Adventure The Lost Dragon Spy
Quest Back to Oz Willy Wonka and the Chocolate Factory Other formats Video games Soundtrack Related William Hanna Joseph Barbera Scott Bradley Hanna - Barbera
Founders William Hanna
</item 3>

<item 4>
List of Tom and Jerry characters
`` Tom '' is a generic name for a male cat . He is usually but not always , portrayed as living a comfortable , or even pampered life , while Jerry ( named ``
Jinx '' in his debut appearance ) is a small , brown , house mouse who always lives in close proximity to Tom . Despite being very energetic , determined and
much larger , Tom is no match for Jerry 's wits . Jerry also possesses surprising strength for his size , approximately the equivalent of Tom 's , lifting items
such as anvils with relative ease and withstanding considerable impacts . Although cats typically chase mice to consume them , it is quite rare for Tom to
actually try to consume Jerry . Most of his attempts are just to torment or humiliate Jerry , sometimes in revenge , and sometimes to obtain a reward from a
human for catching Jerry . By the final `` fade - out '' of each cartoon , Jerry usually emerges triumphant , while Tom is shown as the loser . However , other
results may be reached . On rare occasions , Tom triumphs , usually when Jerry becomes the aggressor or when he pushes Tom a little too far .
</item 4>

<item 5>
Graph query row:
character: Spike the Bulldog
species: None
appearance: None
description: None
</item 5>

<item 6>
List of Tom and Jerry characters
The most frequent love interest of Tom 's is Toodles Galore , who never has any dialogue in the cartoons . Despite five shorts ending with a depiction of Tom 's
apparent death , his demise is never permanent ; he even reads about his own death in a flashback in Jerry 's Diary . He appears to die in explosions in Mouse
Trouble ( after which he is seen in heaven ) , Yankee Doodle Mouse and in Safety Second , while in The Two Mouseketeers he is guillotined offscreen . Hanna -
Barbera era ( edit ) The following characters were introduced in the theatrical shorts that were directed by William Hanna and Joseph Barbera . Spike and Tyke (
edit ) Main article : Spike and Tyke ( characters ) Spike , occasionally referred to as Butch or Killer , is a stern but occasionally dumb American bulldog who
is particularly disapproving of cats , but a softie when it comes to mice ( though in his debut appearance , Dog Trouble , Spike goes after both Tom and Jerry )
, and later , his son Tyke .
</item 6>

<item 7>
Graph query row:
character: Butch
species: None
appearance: None
description: None
</item 7>

<item 8>
List of Tom and Jerry characters
Trojan Dog ' in Tom and Jerry Comedy Show ) and movies thereafter . In his first appearance , the mad scientist is the owner of Jerry ( who is also his
assistant ) and has numerous cats trapped in a prison . When Jerry picks an orange cat named Lightning , the other cats flee in terror as the scientist pulls
him out for his latest experiment . He switches the mind of the orange cat with that of a bulldog and presents him as his present to Jerry . Thin lady ( edit )
An unnamed thin white lady became Tom 's owner in Buddies Thicker Than Water . Gene Deitch opted not to use Mammy 's character in his shorts , as he felt a ``
stereotypical black housekeep '' character `` did n't work in a modern context . '' Chuck Jones would also use her in The Unshrinkable Jerry Mouse and in the
re-animated version of Saturday Evening Puss . She also appeared in the movie Tom and Jerry : The Fast and the Furry . Like many other early human characters ,
her face was never shown .
</item 8>

<item 9>
List of Tom and Jerry characters
Butch is usually decipted as a homeless alley / street cat although in the 1956 short , Blue Cat Blues , Butch is apparently a millionaire who wins the
attention of Tom 's love interest due to his immense wealth much to Tom 's dismay . Butch also appears in the Spike and Tyke cartoon Scat Cats as a house cat
owned by George and Joan who in the original series own Tom . Toodles Galore ( edit ) Toodles Galore is an attractive white female cat , and is supposedly Tom
's usual love interest , although Tom is a reputed playboy , and had other love interests before and after Toodles . Toodles is the only love interest who
appeared more than twice , and is probably the most favored . During the classic era , Tom had to compete twice against Butch and even once against Spike for
Toodles 's affection , and he lost them all . In Casanova Cat , Toodles fell in love with Jerry . However , in her final short Love Me , Love My Mouse , she
remained with Tom .
</item 9>

<item 10>
List of Tom and Jerry characters
In Tom and Jerry : Spy Quest , they are seen working for Doctor Zin . They also make an appearance in Tom and Jerry : Willy Wonka and the Chocolate Factory .
They were named after the Warner Bros. 1943 short Tin Pan Alley Cats . Tin is voiced by Greg Ellis , Pan by Jess Harnell and Alley by Richard McGonagle . Butch
( dog ) ( edit ) Main article : Butch Dog ( Tex Avery ) Butch ( Originally known as Spike , until renamed as Butch to avoid confusion with Spike from the Tom
and Jerry cartoons ) is the name of another bulldog from Tex Avery 's shorts in the 1940s and 1950s . He was one of the main antagonists in Droopy shorts (
although he was replaced by Hanna - Barbera 's Spike in the 1980s ) and also had a series of his own . He appears in a number of movies e.g. Tom and Jerry and
the Wizard of Oz and Tom and Jerry : The Magic Ring . References ( edit ) Jump up ^ Deitch , Gene ( 2015 ) .
</item 10>

<item 11>
Graph query row:
character: Spike
species: None
appearance: None
description: None
</item 11>

<item 12>
List of Tom and Jerry characters
This backfired and left Mechano with a black face , while crying in frustration . He sneaks back and promptly shoots Tom , who attempts to shoot Mechano back
for his insubordination . However , the shot came out on the wrong end and Tom faints leaving Mechano laughing himself silly . He also appeared in the Tom and
Jerry Tales episode `` Tin Cat of Tomorrow '' where he is named Verminator 7000 . Mechano ( called `` Robot Cat '' ) also appeared as a fighter in the video
game , War of the Whiskers , and was voiced by Dee Bradley Baker . The St. Bernard ( edit ) The St. Bernard is a dog who first appeared in Puttin ' on the Dog
and then again in The A-Tom - Inable Snowman . Unlike other dogs in the series , he is not hostile to cats and even attends Tom medically . Eagle ( edit ) Eagle
is an Eagle that tries to steal Jerry for food from Tom .
</item 12>

<item 13>
List of Tom and Jerry characters
( as a ticking time bomb set by Tom ) . Toots ( edit ) Toots is the name given to three different characters that have appeared infrequently . The first of
these is a fluffy beige cat who appears in Puss n ' Toots and The Mouse Comes to Dinner , the latter being her only speaking role . She also has a cameo in
Smitten Kitten . This version of Toots also appeared in some 1940s Tom and Jerry comics . Toots is also a different cat by the same name who appears in The Zoot
Cat and in the Tom and Jerry Tales episode `` Kitty Cat Blues . '' She has occasionally and erroneously been referred to as `` Sheikie , '' but this is actually
Tom 's nickname , as seen on a gift card in the cartoon . Finally Toots is the name of a mouse who serves as Jerry 's love interest in Blue Cat Blues . King of
France ( edit ) In some of the mouseketeer series of cartoons , Tom serves the king of France . This is an obvious reference to Louis XVI of France , who is
shown as an extremely short - tempered and cruel monarch .
</item 13>
=== END ITEM id=fc0259790665c268 ===

=== ITEM id=ea074a0015bdbaad ===
Gold sentences and facts:
Sentence 0: The official music video for the song was directed by P.R. Brown and produced by Steve Lamar . The video was filmed along the Californian coast at The Inn at Newport Ranch , a resort and cattle ranch to the north of San Francisco . It features Brown and his duet partner Lauren Alaina singing along the coast . It was released on May 14 , 2017 . An earlier lyric video was first released on May 4 , 2017 , and shows Brown and Alaina cruising down a road .
  Fact 0.0: The official music video for the song was directed by P.R. Brown.
  Fact 0.1: The official music video for the song was produced by Steve Lamar.
  Fact 0.2: The video was filmed along the Californian coast.
  Fact 0.3: The video was filmed at The Inn at Newport Ranch.
  Fact 0.4: The Inn at Newport Ranch is a resort and cattle ranch.
  Fact 0.5: The Inn at Newport Ranch is north of San Francisco.
  Fact 0.6: The music video features P.R. Brown and his duet partner Lauren Alaina singing along the coast.
  Fact 0.7: The official music video was released on May 14, 2017.
  Fact 0.8: An earlier lyric video for the song was first released on May 4, 2017.
  Fact 0.9: The earlier lyric video shows P.R. Brown and Lauren Alaina cruising down a road.

Retrieved items:
<item 1>
What Ifs
It features Brown and his duet partner Lauren Alaina singing along the coast . It was released on May 14 , 2017 . An earlier lyric video was first released on
May 4 , 2017 , and shows Brown and Alaina cruising down a road . Chart performance ( edit ) Weekly charts ( edit ) Chart ( 2017 ) Peak position Canada (
Canadian Hot 100 ) 45 Canada Country ( Billboard ) US Billboard Hot 100 26 US Country Airplay ( Billboard ) US Hot Country Songs ( Billboard ) Year - end charts
( edit ) Chart ( 2017 ) Position US Billboard Country Airplay 7 US Billboard Hot Country Songs US Billboard Hot 100 72 Certifications and sales ( edit ) Region
Certification Certified units / Sales Canada ( Music Canada ) Platinum 80,000 United States ( RIAA ) Platinum 679,000 sales figures based on certification alone
shipments figures based on certification alone References ( edit ) Jump up ^ Dukes , Billy ( January 24 , 2017 ) . `` Kane Brown ( Feat . Lauren Alaina ) , '
What Ifs ' ( Listen ) '' . Taste of Country .
</item 1>

<item 2>
What Ifs
wikipedia.org/w/index.php?title=What_Ifs&oldid=838232791 '' Categories : 2016 songs 2017 singles Kane Brown songs Lauren Alaina songs RCA Records Nashville
singles Vocal duets Song recordings produced by Dann Huff Songs written by Jordan Schmidt Billboard Hot Country Songs number - one singles Hidden categories :
Music infoboxes with deprecated parameters Articles with hAudio microformats Singlechart usages for Canada Singlechart called without song Singlechart usages
for Billboardcanadacountry Singlechart usages for Billboardhot100 Singlechart usages for Billboardcountryairplay Singlechart usages for Billboardcountrysongs
Certification Table Entry usages for Canada Certification Table Entry usages for United States Talk Contents About Wikipedia Español Edit links This page was
last edited on 25 April 2018 , at 18 : 59 . About Wikipedia
</item 2>

<item 3>
What Ifs
Jump up ^ Bjorke , Matt ( January 3 , 2018 ) . `` Top 30 Country Digital Singles Chart : January 3 , 2018 '' . Roughstock . Retrieved January 14 , 2017 . ^ Jump
up to : Bjorke , Matt ( March 6 , 2018 ) . `` The Top 30 Digital Country Singles : March 5 , 2018 '' . Roughstock . Retrieved March 11 , 2018 . Jump up ^ ``
Watch Kane Brown , Lauren Alaina 's Picturesque ' What Ifs ' Video '' . Rolling Stone . May 15 , 2017 . Jump up ^ Kimble , Lindsay ( May 4 , 2017 ) . `` From
Middle School Chorus to Nashville , Kane Brown and Lauren Alaina Team Up for Duet ' What Ifs ' '' . Jump up ^ `` Kane Brown Chart History ( Canadian Hot 100 )
'' . Billboard . Retrieved October 24 , 2017 . Jump up ^ `` Kane Brown Chart History ( Canada Country ) '' . Billboard .
</item 3>

<item 4>
What Ifs
According to Brown , another song was going to be used in the album , however , he was asked so much about `` What Ifs '' on social media that it was then
decided that the song should be included in the album . Composition ( edit ) The song is in the key of G minor with a moderately fast tempo of approximately 126
beats per minute . It follows the chord progression Gm - B ♭ - F. Brown 's vocals range from F - B ♭ . Commercial performance ( edit ) The song sold 6,000
downloads in the US the week the album was released on December 2 , 2016 , allowing it to chart at number 37 on the US Billboard Hot Country Songs chart . The
song first entered the US Billboard Country Airplay chart for the week of March 4 , 2017 at number 60 . It reached No. 1 on Hot Country Songs in October 2017 ,
ending the record breaking 34 - week reign of Sam Hunt 's `` Body Like a Back Road '' . The song is the first No. 1 on the chart for both Brown and Alaina .
</item 4>

<item 5>
What Ifs
Music Canada . Retrieved February 8 , 2018 . Preceded by `` Body Like a Back Road '' by Sam Hunt Billboard Hot Country Songs number - one single October 21 --
November 11 , 2017 Succeeded by `` When It Rains It Pours '' by Luke Combs Preceded by `` All the Pretty Girls '' by Kenny Chesney Billboard Country Airplay
number - one single October 28 , 2017 Kane Brown Studio albums Kane Brown ( 2016 ) EPs Closer ( 2015 ) Chapter 1 ( 2016 ) Singles `` Do n't Go City on Me '' ``
Used to Love You Sober '' `` What Ifs '' `` Heaven '' Lauren Alaina Studio albums Wildflower Road Less Traveled EPs Lauren Alaina Singles `` Like My Mother Does
'' `` Georgia Peaches '' `` Eighteen Inches '' `` Barefoot and Buckwild '' `` Next Boyfriend '' `` Road Less Traveled '' `` Doin ' Fine '' Other singles `` My
Grown Up Christmas List '' `` What Ifs '' Retrieved from `` https://en.wikipedia.org/w/index.
</item 5>

<item 6>
What Ifs
Jump up ^ `` Kane Brown Chart History ( Canada Country ) '' . Billboard . Retrieved November 14 , 2017 . Jump up ^ `` Kane Brown Chart History ( Hot 100 ) '' .
Billboard . Retrieved October 24 , 2017 . Jump up ^ `` Kane Brown Chart History ( Country Airplay ) '' . Billboard . Retrieved October 17 , 2017 . Jump up ^ ``
Kane Brown Chart History ( Hot Country Songs ) '' . Billboard . Retrieved October 10 , 2017 . Jump up ^ `` Country Airplay Songs -- Year - End 2017 '' .
Billboard . Retrieved December 18 , 2017 . Jump up ^ `` Hot Country Songs -- Year - End 2017 '' . Billboard . Retrieved December 18 , 2017 . Jump up ^ `` Hot
100 Songs -- Year - End 2017 '' . Billboard . Retrieved December 12 , 2017 . Jump up ^ `` Canadian single certifications -- Kane Brown -- What Ifs '' . Music
Canada .
</item 6>

<item 7>
What Ifs
The song is the first No. 1 on the chart for both Brown and Alaina . The following week , for charts dated October 28 , 2017 , `` What Ifs '' stayed at number
one , as well as topping the Country Airplay and Country Streaming Songs charts . Together with the release of the deluxe edition of the album which also
reached number on Top Country Albums , and another number one for a new track `` Heaven '' on digital sales , Brown became the first artist to have simultaneous
number ones on all five main country charts . The song was certified Platinum by the RIAA on October 2 , 2017 . It was the third most - consumed country song of
2017 , with 1,551,000 including streams ( 599,000 in pure sale ) sold in the year . It has sold 679,000 copies in the US as of March 2018 . Music video ( edit )
The official music video for the song was directed by P.R. Brown and produced by Steve Lamar . The video was filmed along the Californian coast at The Inn at
Newport Ranch , a resort and cattle ranch to the north of San Francisco .
</item 7>

<item 8>
Graph query row:
work: official music video for What Ifs
relationship: LOCATED_IN
location: The Inn at Newport Ranch
</item 8>

<item 9>
What Ifs
The song was released with the album through RCA Nashville and was serviced to radio as the third single on February 6 , 2017 . Contents ( hide ) 1 Background 2
Composition 3 Commercial performance 4 Music video 5 Chart performance 5.1 Weekly charts 5.2 Year - end charts 6 Certifications and sales 7 References
Background ( edit ) The song was written by Brown with Matthew McGinn and Jordan Schmidt . The song is about a couple where the question `` what if '' is asked
repeatedly about what would happen in their relationship . It features a duet with Lauren Alaina , a school friend from Lakeview - Fort Oglethorpe High School
when he was living in Fort Oglethorpe , Georgia . According to Brown , the song was initially not intended to be a duet , he said : `` Once we decided on adding
a woman to `` What Ifs , '' I knew that I wanted Lauren to sing it , and , of course , she killed it . ''
</item 9>

<item 10>
What Ifs
What Ifs - wikipedia What Ifs Jump to : navigation , search `` What Ifs '' Single by Kane Brown featuring Lauren Alaina from the album Kane Brown Released
February 6 , 2017 ( 2017 - 02 - 06 ) Format Digital download Genre Country Length 3 : 08 Label RCA Nashville Songwriter ( s ) Kane Brown Matthew McGinn Jordan
Schmidt Producer ( s ) Dann Huff Kane Brown singles chronology `` Thunder in the Rain '' ( 2016 ) `` What Ifs '' ( 2017 ) `` Heaven '' ( 2017 ) `` Thunder in
the Rain '' ( 2016 ) `` What Ifs '' ( 2017 ) `` Heaven '' ( 2017 ) Lauren Alaina singles chronology `` Road Less Traveled '' ( 2016 ) Road Less Traveled 2016 ``
What Ifs '' ( 2017 ) What Ifs2017 `` Doin ' Fine '' ( 2017 ) Doin ' Fine 2017 `` What Ifs '' is a song recorded by American country music singer Kane Brown
featuring Lauren Alaina for his self - titled debut album .
</item 10>

<item 11>
What Ifs
Taste of Country . ^ Jump up to : Konicki , Lisa ( February 8 , 2017 ) . `` Kane Brown 's Current Single , `` What Ifs , '' With Lauren Alaina Almost Did n't
Make The Album `` . Nash Country Daily . Jump up ^ Freeman , Jon ( December 2 , 2016 ) . `` Kane Brown on Personal New Album , Why He Wo n't Sing First Hit :
Ram Report '' . Rolling Stone . ^ Jump up to : Asker , Jim ( October 17 , 2017 ) . `` Kane Brown Becomes First Artist To Simultaneously Lead Five Country Charts
'' . Billboard . Jump up ^ `` ' What Ifs ' sheet music '' . MusicNotes.com . Retrieved 5 February 2018 . Jump up ^ Bjorke , Matt ( December 13 , 2016 ) . `` Top
30 Digital Country Singles Chart : December 13 , 2016 '' . Roughstock . Retrieved February 5 , 2017 . Jump up ^ `` Country Airplay '' . Billboard .
</item 11>
=== END ITEM id=ea074a0015bdbaad ===

=== ITEM id=98e5bea45bb05cb9 ===
Gold sentences and facts:
Sentence 0: Nigeria is the largest oil and gas producer in Africa . Crude oil from the delta basin comes in two types : light , and comparatively heavy -- the lighter around 36 gravity and the heavier , 20 -- 25 gravity . Both types are paraffinic and low in sulfur .
  Fact 0.0: Nigeria is the largest oil and gas producer in Africa.
  Fact 0.1: Crude oil from the delta basin comes in two types: light and comparatively heavy.
  Fact 0.2: The lighter crude oil from the delta basin is around 36 gravity.
  Fact 0.3: The heavier crude oil from the delta basin is 20 to 25 gravity.
  Fact 0.4: Both types of crude oil from the delta basin are paraffinic.
  Fact 0.5: Both types of crude oil from the delta basin are low in sulfur.

Retrieved items:
<item 1>
Petroleum industry in Nigeria
Mobil operates primarily out of the Qua Iboe Terminal in Akwa Ibom State , while Chevron owns the Escravos Terminal located in Delta State and has a storage
capacity of 3.6 million barrels ( 570,000 m ) . Agip operates the Brass Terminal in Brass , a town 113 kilometres ( 70 miles ) southwest of Port Harcourt and
has a storage capacity of 3,558,000 barrels ( 565,700 m ) . Texaco operates the Pennington Terminal . Offshore ( edit ) Oil companies in Africa investigate
offshore production as an alternative area of production . Deepwater production mainly involves underwater drilling that exists 400 metres ( 1,300 ft ) or more
below the surface of the water . By expanding to deep water drilling the possible sources for finding new oil reserves is expanded . Through the introduction of
deep water drilling 50 % more oil is extracted than before the new forms of retrieving the oil . Angola and Nigeria are the largest oil producers in Africa . In
Nigeria , the deepwater sector still has a large avenue to expand and develop .
</item 1>

<item 2>
Graph query row:
region: North
resource: oil revenues
association: None
</item 2>

<item 3>
Petroleum industry in Nigeria
The main reasons underlying this trend are that some Gulf Coast refiners have reduced Nigerian imports in favor of domestically - produced crude , and that two
refineries in the U.S. East Coast , which were significant buyers of Nigerian crude , were idled in late 2011 . As a result , Nigerian crude as a share of total
United States imports has fallen to 5 percent in the first half of 2012 , down from 10 and 11 percent in the first half of 2011 and 2010 , respectively ,
according to the International Energy Agency . According to the CIA World Factbook , Nigeria 's main export partners are the United States , India , Brazil ,
Spain , France and the Netherlands . Shell has been working in Nigeria since 1936 , and currently dominates gas production in the country , as the Niger Delta ,
which contains most of Nigeria 's gas resources , also houses most of Shell 's hydrocarbon assets . History and politics ( edit ) See also : History of Nigeria
and Niger Delta An Ethno - linguistic map of Nigeria .
</item 3>

<item 4>
Graph query row:
region: Europe
resource: palm oil
association: None
</item 4>

<item 5>
Petroleum industry in Nigeria
Though Nigeria is a major oil exporter , it imports most of its gasoline , and when fuel subsidies were lifted in January 2012 , fuel increased from roughly
$1.70 per gallon to $3.50 . Nigeria produces a form of oil ideal for the United States , has huge reserves , and has increased its production to 2.8 million
barrels ( 450,000 m ) of oil a day . But this , some say , is all a resource curse that is hurting Nigeria and disadvantaging her people . Oil theft ( edit ) A
report analysing the effect of oil theft in Nigeria revealed in July 2013 that Nigeria lost $10.9 billion in potential oil revenues between 2009 and 2011 .
Environmental impact ( edit ) See also : Environmental issues in Nigeria Map of vegetation in Nigeria The Niger Delta comprises 70,000 square kilometres (
27,000 square miles ) of wetlands formed primarily by sediment deposition . Home to more than 20 million people and 40 different ethnic groups , this floodplain
makes up 7.5 % of Nigeria 's total land mass . It is the largest wetland and maintains the third - largest drainage area in Africa .
</item 5>

<item 6>
Graph query row:
region: delta basin
resource: crude oil
association: None
</item 6>

<item 7>
Petroleum industry in Nigeria
film era New Wave Coat of arms Cuisine Demonyms Flag List of Nigerians Literature Media Newspapers Radio TV Music Public holidays Religion Smoking Sport Video
gaming Outline Index Portal Petroleum industry in Africa Sovereign states Algeria Angola Benin Botswana Burkina Faso Burundi Cameroon Cape Verde ( Cabo Verde )
Central African Republic Chad Comoros Democratic Republic of the Congo Republic of the Congo Djibouti Egypt Equatorial Guinea Eritrea Ethiopia Gabon The Gambia
Ghana Guinea Guinea - Bissau Ivory Coast ( Côte d'Ivoire ) Kenya Lesotho Liberia Libya Madagascar Malawi Mali Mauritania Mauritius Morocco Mozambique Namibia
Niger Nigeria Rwanda São Tomé and Príncipe Senegal Seychelles Sierra Leone Somalia South Africa South Sudan Sudan Swaziland Tanzania Togo Tunisia Uganda Zambia
Zimbabwe States with limited recognition Sahrawi Arab Democratic Republic Somaliland Dependencies and other territories Canary Islands / Ceuta / Melilla ( Spain
) Madeira ( Portugal ) Mayotte / Réunion ( France ) Saint Helena / Ascension Island / Tristan da Cunha ( United Kingdom ) Retrieved from ``
https://en.wikipedia.org/w/index.
</item 7>

<item 8>
Petroleum industry in Nigeria
An open - air market for illegal crude oil operates off the Niger Delta , called the Togo Triangle . Natural gas ( edit ) Natural gas reserves are well over
5,300 km ( 187 × 10 ^ cu ft ) , the gas reserves are three times as substantial as the crude oil reserves . The biggest natural gas initiative is the Nigerian
Liquified Natural Gas Company , which is operated jointly by several companies and the state . It began exploration and production in 1999 . Chevron is also
attempting to create the Escravos Gas Utilization project which will be capable of producing 4,500,000 m ( 160 × 10 ^ cu ft ) per day . gas reserves . In 2008 ,
the government prepared a Gas Master Plan that was intended to promote natural gas production and encourage the supply of natural gas to domestic power stations
so as to help alleviate the country 's electricity shortages . There is also an export gas pipeline , known as the West African Gas Pipeline , in the works but
has encountered numerous setbacks . The pipeline would allow for transportation of natural gas to Benin , Ghana , Togo , and Cote d'Ivoire .
</item 8>

<item 9>
Petroleum industry in Nigeria
Petroleum industry in Nigeria - wikipedia Petroleum industry in Nigeria Jump to : navigation , search Nigeria is the largest oil and gas producer in Africa .
Crude oil from the delta basin comes in two types : light , and comparatively heavy -- the lighter around 36 gravity and the heavier , 20 -- 25 gravity . Both
types are paraffinic and low in sulfur . The gates of the oil refinery in Port Harcourt Contents ( hide ) 1 History of oil exploration 2 Production and
exploration 2.1 Offshore 2.2 Natural gas 2.3 Downstream 3 Current manufacturing 4 History and politics 4.1 Colonial legacy ( 1800s -- 1960s ) 4.2 Implications
and causes of civil war ( 1966 -- 1970 ) 4.3 Industry nationalisation ( 1970 -- 1979 ) 4.4 Attempted democracy and debt ( 1979 -- 1983 ) 4.5 Return to military
rule and electoral annulment ( 1983 -- 1993 ) 4.6 An environment of crisis ( 1993 -- present ) 5 Operating agreements 5.
</item 9>

<item 10>
Petroleum industry in Nigeria
Offshore spills , which are usually much greater in scale , contaminate coastal environments and cause a decline in local fishing production . Nigerian
regulations are weak and rarely enforced allowing oil companies , in essence , to self - regulate . Natural gas flaring ( edit ) See also main article :
Environmental issues in the Niger Delta Nigeria flares more natural gas associated with oil extraction than any other country , with estimates suggesting that
of the 99,000,000 m ( 3.5 × 10 ^ cu ft ) of associated gas ( AG ) produced annually , 71,000,000 m ( 2.5 × 10 ^ cu ft ) , or about 70 % is wasted via flaring .
Statistical data associated with gas flaring is notoriously unreliable , but AG wasted during flaring is estimated to cost Nigeria US $2.5 billion on a yearly
basis . Companies operating in Nigeria harvest natural gas for commercial purposes , however prefer to extract it gas from deposits where it is found in
isolation as non-associated gas . It is costly to separate commercially viable associated gas from oil , hence gas flaring to increase crude production .
</item 10>

<item 11>
Graph query row:
region: Africa
resource: Petroleum industry in Africa
association: None
</item 11>

<item 12>
Graph query row:
region: refineries
resource: crude oil
association: None
</item 12>

<item 13>
Graph query row:
region: Agbami oilfields
resource: Agbami oilfields hit full production
association: None
</item 13>

<item 14>
Graph query row:
region: Forcados Terminal
resource: Shell
association: Shell owns the Forcados Terminal
</item 14>

<item 15>
Graph query row:
region: west
resource: oil revenues
association: None
</item 15>

<item 16>
Petroleum industry in Nigeria
Towards the end of the 1950s , non-British firms were granted license to explore for oil : Mobil in 1955 , Tenneco in 1960 , Gulf Oil and later Chevron in 1961
, Agip in 1962 , and Elf in 1962 . Prior to the discovery of oil , Nigeria ( like many other African countries ) strongly relied on agricultural exports to
other countries to supply their economy . Many Nigerians thought the developers were looking for palm oil . But after nearly 50 years searching for oil in the
country , Shell - BP discovered the oil at Oloibiri in the Niger Delta . The first oil field began production in 1958 . After that , the economy of Nigeria
should have seemingly have experienced a strong increase . However , competition for the profits from oil created a great level of terror and conflict for those
living in the region . Many citizens of Nigeria believe that they have n't been able to see the economic benefits of oil companies in the state .
</item 16>

<item 17>
Petroleum industry in Nigeria
Retrieved 4 December 2012 . Retrieved 26 November 2012 Jump up ^ U.S. Energy Information Administration ( U.S. EIA ) , `` Nigeria Country Analysis Brief , ''
December 1997 . Jump up ^ http://www.oilandgascouncil.com/expert_insight_articles/review-nigeria-oil-gas-industry Jump up ^ Environmental Resources Managers Ltd
, Niger Delta Environmental Survey Final Report Phase I ; Volume I : Environmental and Socio - Economic Characteristics ( Lagos : Niger Delta Environmental
Survey , September 1997 ) ^ Jump up to : Nigeria : The Political Economy of Oil ISBN 0 - 19 - 730014 - 6 ( Khan , Ahmad ) Jump up ^ US Shuts its Door on Nigeria
's Oil Exports ( ThisDayLive.com ) ^ Jump up to : Nigerian Crude and Gas Industry ( NigeriaBusinessInfo.com ) ^ Jump up to : McLennan , James ; Stewart Williams
( Feb 2005 ) . Deepwater Africa reaches turning point . Oil & Gas Journal . Vol. 103 , 6 ; ABI / INFORM Global . pg. 18 Jump up ^ Khadija Sharife .
</item 17>

<item 18>
Petroleum industry in Nigeria
The most productive region of the nation is the coastal Niger Delta Basin in the Niger Delta or `` South - south '' region which encompasses 78 of the 159 oil
fields . Most of Nigeria 's oil fields are small and scattered , and as of 1990 , these small unproductive fields accounted for 62.1 % of all Nigerian
production . This contrasts with the sixteen largest fields which produced 37.9 % of Nigeria 's petroleum at that time . As a result of the numerous small
fields , an extensive and well - developed pipeline network has been engineered to transport the crude oil . Also because of the lack of highly productive
fields , money from the jointly operated ( with the federal government ) companies is constantly directed towards petroleum exploration and production . Nigeria
's petroleum is classified mostly as `` light '' and `` sweet '' , as the oil is largely free of sulphur . Nigeria is the largest producer of sweet oil in OPEC
. This sweet oil is similar in composition to petroleum extracted from the North Sea . This crude oil is known as `` Bonny light '' .
</item 18>

<item 19>
Graph query row:
region: the southeast
resource: Shell
association: None
</item 19>

<item 20>
Graph query row:
region: India
resource: Nigerian oil
association: India is now the largest consumer of Nigerian oil
</item 20>
=== END ITEM id=98e5bea45bb05cb9 ===

=== ITEM id=094e6fe4a99e7f13 ===
Gold sentences and facts:
Sentence 0: Barbara Eden as Stella Johnson , a widowed cosmetics salesperson with a young daughter , whose joie de vivre , her principles , and her individuality anger most of the powerful people in town and on the Parent Teachers Association board . She inevitably becomes PTA president after embarrassing Flora , the snobby former President and her equally snobby friends . Ronny Cox as Willis `` Will '' Newton , a wealthy member of the PTA who does n't agree with Flora and her snobby allies and is in love with Stella . He and Stella get married at the end of the movie . Nanette Fabray as Alice Finley , Stella 's best friend and co-conspirator against the snobs of Harper Valley . She owns and operates the La Moderne Beauty Shop . Louis Nye as Kirby Baker , a local real estate agent who is allied with Flora and her friends . Notorious for being a lecherous playboy and infamous for impregnating his secretaries , one of whom , Gladys Wilmot , left town to see an abortionist in Cincinnati . As part of a plan to discredit Stella , he tries to foreclose on Stella 's house , which infuriates her . He then gets arrested and jailed for attacking Myrna Wong , a friend of Stella 's , who rebuffs his advances , throwing him by use of martial arts and destroying the office . John Fiedler as Bobby Taylor , a man on the PTA who lusts after Stella . Overly dominated by his wife , Holly . Although open to allow Stella to speak , he is not above having a fellow PTA board member kidnapped to commit election fraud . He was arrested for indecent exposure after being set up by Stella . Ron Masak as Herbie Maddox , the head bartender at Kelly 's bar , and Stella 's ally against Flora and her friends . He has a crush on Stella , and is one of her friends . Clint Howard as Corley , a young delivery person , who works for the local florist shop , who brings Stella flowers from Will . Susan Swift as Dee Johnson , Stella 's 14 - year - old daughter , who is threatened with expulsion from school if her mother does n't conform to the way the PTA wants her to behave . She has a hand in assisting her mother and Alice with one of their revenge schemes . She loses her braces , gets a makeover , and ends up with a boyfriend , Carlyle . Pat Paulsen as Otis Harper , the descendant of the founding family of Harper Valley . A PTA member who is allied with Flora and her friends . Has a tendency to drink all the time , which annoys his wife , Bertha . He had a stampede of elephants , painted pink , stormed through his house , destroying it . Amzie Strickland as Shirley Thompson , a member of the PTA board who tends to drink a lot of gin . Though she is allied with Flora and her friends , she is one of four PTA members ( Will , Skeeter and Holly Taylor are the others ) who are spared any retribution from Stella . She was seen as the most sympathetic of the board , as she tended to be more manipulated as opposed to being manipulative . It was never revealed whether or not she retained her seat on the PTA , but it is presumed that she became more sympathetic towards Stella as she was seen waving goodbye at Olive as she was being arrested . Bob Hastings as Skeeter Duggan , a member of the PTA board , and the town 's Notary public who is sympathetic to Stella . He is kidnapped by Dutch and Tex at Flora Simpson Reilly 's orders . Audrey Christie as Flora Simpson - Reilly , The movie 's main antagonist . She is the pompous and extremely snobbish president of the PTA board who thoroughly despises Stella ( who defeats her for president at the end of the movie ) and all she stands for . So desperate is she to retain her power and prestige in the community that she has a fellow member of the PTA ( Skeeter Duggan ) kidnapped so she and her allies can commit election fraud . She is embarrassed when her permanent set is destroyed by a hair depilatory at the beauty salon . Molly Dodd as Olive Glover , the recording secretary of the PTA who is a gambling addict . Clearly the most corrupt and most vindictive member of the PTA board , she is eventually arrested for embezzlement after having a load of hay and horse manure dumped on top of her . She is the one who initially angers Stella by sending her the letter denouncing her , and tries to get Dee 's friend , Mavis , arrested for embezzlement . Fay DeWitt as Willamae Jones , a snide biology teacher at the junior high school who is also a member of the PTA . She makes no secret of the fact that she favors the Reilly twins , due to the connections between her and Flora . Notorious for being a nymphomaniac ( though she hides it under a prim - and - proper façade ) , she is caught in flagrante delicto with Barney Crunk , the town 's mailman , on film . DeVera Marcus as Holly Taylor , Bobby 's obviously overbearing wife , who tends to drink a lot when Bobby is out of town , and does n't approve of Bobby 's lusting , or of anything else he does ( this is different from the character in the song , who `` seems to use a lot of ice whenever ( her husband ) 's away '' , implying adultery with the iceman ) . However , despite her ties to Flora and her scheming friends , she , like Shirley Thompson , is spared any retribution from Stella , due to Stella refusing to date Bobby . Louise Foley as Mavis , Dee Johnson 's best friend , who is nearly framed for a crime she did not commit by the vindictive Olive Glover . Brian Cook as Carlyle , a popular track star who chooses Dee , after she gets a makeover , and angers Bettina Reilly in the process . Laura Teige as Bettina Reilly , One of Flora 's twin granddaughters , who wants to be with Carlyle , and is jealous when he chooses Dee over her . She tends to be as snotty as her grandmother . Jan Teige as Edwina Reilly , Bettina 's twin sister , who is just as snotty as her . Irene Yah - Ling Sun as Myrna Wong , a friend of Stella 's who is a judo expert ; she is assaulted by Kirby Baker ( after leading him on ; she destroys him and his office in retaliation ) and has him sent to jail for assault . Tobias Anderson as `` Cranky '' Barney Crunk , the town 's nosy mailman who had the bad habit of peeking in everyone 's mail . He is caught in flagrante delicto on film with Willamae Jones . Royce D. Applegate as Dutch , one of two kidnappers who abduct Skeeter Duggan at Flora Simpson - Reilly 's orders . J.J. Barry as Tex , the mastermind of the kidnap plot to get Skeeter Duggan out of the way so Flora Simpson - Reilly and her friends could commit election fraud . He and his accomplice Dutch are arrested and the two of them rat out Flora and her friends on their scheme . Pitt Herbert as Henry Reilly , Flora 's overly dominated husband and grandfather to Edwina and Bettina . Henry is wealthy , but is very weak - willed and is known for acquiescing to his wife 's schemes . Arlen Stuart as Bertha Harper , wife of Otis Harper , who gets annoyed by her husband 's continual drinking . She is terrified when her house is destroyed by pink - painted elephants .
  Fact 0.0: Barbara Eden plays Stella Johnson.
  Fact 0.1: Stella Johnson is a widowed cosmetics salesperson.
  Fact 0.2: Stella Johnson has a young daughter.
  Fact 0.3: Stella Johnson's joie de vivre angers most of the powerful people in town.
  Fact 0.4: Stella Johnson's joie de vivre angers most of the powerful people on the Parent Teachers Association board.
  Fact 0.5: Stella Johnson's principles anger most of the powerful people in town.
  Fact 0.6: Stella Johnson's principles anger most of the powerful people on the Parent Teachers Association board.
  Fact 0.7: Stella Johnson's individuality angers most of the powerful people in town.
  Fact 0.8: Stella Johnson's individuality angers most of the powerful people on the Parent Teachers Association board.
  Fact 0.9: Stella Johnson inevitably becomes PTA president.
  Fact 0.10: Stella Johnson becomes PTA president after embarrassing Flora.
  Fact 0.11: Flora is the snobby former President of the PTA.
  Fact 0.12: Stella Johnson embarrasses Flora and Flora's equally snobby friends.
  Fact 0.13: Ronny Cox plays Willis 'Will' Newton.
  Fact 0.14: Will Newton is a wealthy member of the PTA.
  Fact 0.15: Will Newton does not agree with Flora and her snobby allies.
  Fact 0.16: Will Newton is in love with Stella Johnson.
  Fact 0.17: Will Newton and Stella Johnson get married at the end of the movie.
  Fact 0.18: Nanette Fabray plays Alice Finley.
  Fact 0.19: Alice Finley is Stella Johnson's best friend.
  Fact 0.20: Alice Finley is Stella Johnson's co-conspirator against the snobs of Harper Valley.
  Fact 0.21: Alice Finley owns and operates the La Moderne Beauty Shop.
  Fact 0.22: Louis Nye plays Kirby Baker.
  Fact 0.23: Kirby Baker is a local real estate agent.
  Fact 0.24: Kirby Baker is allied with Flora and her friends.
  Fact 0.25: Kirby Baker is notorious for being a lecherous playboy.
  Fact 0.26: Kirby Baker is infamous for impregnating his secretaries.
  Fact 0.27: Gladys Wilmot is one of Kirby Baker's secretaries.
  Fact 0.28: Gladys Wilmot left town to see an abortionist in Cincinnati.
  Fact 0.29: As part of a plan to discredit Stella Johnson, Kirby Baker tries to foreclose on Stella Johnson's house.
  Fact 0.30: Kirby Baker's attempt to foreclose on Stella Johnson's house infuriates Stella Johnson.
  Fact 0.31: Kirby Baker gets arrested and jailed for attacking Myrna Wong.
  Fact 0.32: Myrna Wong is a friend of Stella Johnson.
  Fact 0.33: Myrna Wong rebuffs Kirby Baker's advances.
  Fact 0.34: Myrna Wong throws Kirby Baker by use of martial arts.
  Fact 0.35: Myrna Wong destroys Kirby Baker's office.
  Fact 0.36: John Fiedler plays Bobby Taylor.
  Fact 0.37: Bobby Taylor is a man on the PTA.
  Fact 0.38: Bobby Taylor lusts after Stella Johnson.
  Fact 0.39: Bobby Taylor is overly dominated by his wife, Holly Taylor.
  Fact 0.40: Bobby Taylor is open to allowing Stella Johnson to speak.
  Fact 0.41: Bobby Taylor is not above having a fellow PTA board member kidnapped to commit election fraud.
  Fact 0.42: Bobby Taylor was arrested for indecent exposure after being set up by Stella Johnson.
  Fact 0.43: Ron Masak plays Herbie Maddox.
  Fact 0.44: Herbie Maddox is the head bartender at Kelly's bar.
  Fact 0.45: Herbie Maddox is Stella Johnson's ally against Flora and her friends.
  Fact 0.46: Herbie Maddox has a crush on Stella Johnson.
  Fact 0.47: Herbie Maddox is one of Stella Johnson's friends.
  Fact 0.48: Clint Howard plays Corley.
  Fact 0.49: Corley is a young delivery person.
  Fact 0.50: Corley works for the local florist shop.
  Fact 0.51: Corley brings Stella Johnson flowers from Will Newton.
  Fact 0.52: Susan Swift plays Dee Johnson.
  Fact 0.53: Dee Johnson is Stella Johnson's 14-year-old daughter.
  Fact 0.54: Dee Johnson is threatened with expulsion from school if her mother does not conform to the way the PTA wants her to behave.
  Fact 0.55: Dee Johnson has a hand in assisting her mother Stella Johnson and Alice Finley with one of their revenge schemes.
  Fact 0.56: Dee Johnson loses her braces.
  Fact 0.57: Dee Johnson gets a makeover.
  Fact 0.58: Dee Johnson ends up with a boyfriend, Carlyle.
  Fact 0.59: Pat Paulsen plays Otis Harper.
  Fact 0.60: Otis Harper is the descendant of the founding family of Harper Valley.
  Fact 0.61: Otis Harper is a PTA member who is allied with Flora and her friends.
  Fact 0.62: Otis Harper has a tendency to drink all the time.
  Fact 0.63: Otis Harper's drinking annoys his wife, Bertha Harper.
  Fact 0.64: A stampede of elephants, painted pink, stormed through Otis Harper's house.
  Fact 0.65: The stampede of pink elephants destroyed Otis Harper's house.
  Fact 0.66: Amzie Strickland plays Shirley Thompson.
  Fact 0.67: Shirley Thompson is a member of the PTA board.
  Fact 0.68: Shirley Thompson tends to drink a lot of gin.
  Fact 0.69: Shirley Thompson is allied with Flora and her friends.
  Fact 0.70: Shirley Thompson is one of four PTA members spared any retribution from Stella Johnson.
  Fact 0.71: The other three PTA members spared any retribution from Stella Johnson are Will, Skeeter, and Holly Taylor.
  Fact 0.72: Shirley Thompson was seen as the most sympathetic member of the PTA board.
  Fact 0.73: Shirley Thompson tended to be more manipulated than manipulative.
  Fact 0.74: It was never revealed whether Shirley Thompson retained her seat on the PTA.
  Fact 0.75: It is presumed that Shirley Thompson became more sympathetic towards Stella Johnson.
  Fact 0.76: Shirley Thompson was seen waving goodbye at Olive as Olive was being arrested.
  Fact 0.77: Bob Hastings plays Skeeter Duggan.
  Fact 0.78: Skeeter Duggan is a member of the PTA board.
  Fact 0.79: Skeeter Duggan is the town's Notary public.
  Fact 0.80: Skeeter Duggan is sympathetic to Stella Johnson.
  Fact 0.81: Skeeter Duggan is kidnapped by Dutch and Tex at Flora Simpson-Reilly's orders.
  Fact 0.82: Audrey Christie plays Flora Simpson-Reilly.
  Fact 0.83: Flora Simpson-Reilly is the movie's main antagonist.
  Fact 0.84: Flora Simpson-Reilly is the pompous and extremely snobbish president of the PTA board.
  Fact 0.85: Flora Simpson-Reilly thoroughly despises Stella Johnson.
  Fact 0.86: Flora Simpson-Reilly despises all that Stella Johnson stands for.
  Fact 0.87: Stella Johnson defeats Flora Simpson-Reilly for president at the end of the movie.
  Fact 0.88: Flora Simpson-Reilly is desperate to retain her power and prestige in the community.
  Fact 0.89: Flora Simpson-Reilly has Skeeter Duggan, a fellow PTA member, kidnapped so that she and her allies can commit election fraud.
  Fact 0.90: Flora Simpson-Reilly is embarrassed when her permanent set is destroyed by a hair depilatory at the beauty salon.
  Fact 0.91: Molly Dodd plays Olive Glover.
  Fact 0.92: Olive Glover is the recording secretary of the PTA.
  Fact 0.93: Olive Glover is a gambling addict.
  Fact 0.94: Olive Glover is clearly the most corrupt member of the PTA board.
  Fact 0.95: Olive Glover is clearly the most vindictive member of the PTA board.
  Fact 0.96: Olive Glover has a load of hay and horse manure dumped on top of her.
  Fact 0.97: Olive Glover is eventually arrested for embezzlement, after the hay and manure are dumped on her.
  Fact 0.98: Olive Glover initially angers Stella Johnson by sending her a letter denouncing her.
  Fact 0.99: Olive Glover tries to get Mavis arrested for embezzlement.
  Fact 0.100: Mavis is Dee Johnson's friend.
  Fact 0.101: Fay DeWitt plays Willamae Jones.
  Fact 0.102: Willamae Jones is a snide biology teacher at the junior high school.
  Fact 0.103: Willamae Jones is also a member of the PTA.
  Fact 0.104: Willamae Jones makes no secret that she favors the Reilly twins.
  Fact 0.105: Willamae Jones favors the Reilly twins due to the connections between her and Flora.
  Fact 0.106: Willamae Jones is notorious for being a nymphomaniac.
  Fact 0.107: Willamae Jones hides her nymphomania under a prim-and-proper facade.
  Fact 0.108: Willamae Jones is caught in flagrante delicto with Barney Crunk on film.
  Fact 0.109: Barney Crunk is the town's mailman.
  Fact 0.110: DeVera Marcus plays Holly Taylor.
  Fact 0.111: Holly Taylor is Bobby Taylor's obviously overbearing wife.
  Fact 0.112: Holly Taylor tends to drink a lot when Bobby Taylor is out of town.
  Fact 0.113: Holly Taylor does not approve of Bobby Taylor's lusting.
  Fact 0.114: Holly Taylor does not approve of anything else Bobby Taylor does.
  Fact 0.115: The Holly Taylor character in the movie is different from the character in the song.
  Fact 0.116: The character in the song 'seems to use a lot of ice whenever (her husband)'s away', implying adultery with the iceman.
  Fact 0.117: Holly Taylor has ties to Flora and her scheming friends.
  Fact 0.118: Holly Taylor, like Shirley Thompson, is spared any retribution from Stella Johnson.
  Fact 0.119: Holly Taylor is spared retribution because Stella Johnson refuses to date Bobby Taylor.
  Fact 0.120: Louise Foley plays Mavis.
  Fact 0.121: Mavis is Dee Johnson's best friend.
  Fact 0.122: Mavis is nearly framed for a crime she did not commit by Olive Glover.
  Fact 0.123: Olive Glover is vindictive.
  Fact 0.124: Brian Cook plays Carlyle.
  Fact 0.125: Carlyle is a popular track star.
  Fact 0.126: Carlyle chooses Dee Johnson after she gets a makeover.
  Fact 0.127: Carlyle's choice of Dee Johnson angers Bettina Reilly.
  Fact 0.128: Laura Teige plays Bettina Reilly.
  Fact 0.129: Bettina Reilly is one of Flora's twin granddaughters.
  Fact 0.130: Bettina Reilly wants to be with Carlyle.
  Fact 0.131: Bettina Reilly is jealous when Carlyle chooses Dee over her.
  Fact 0.132: Bettina Reilly tends to be as snotty as her grandmother, Flora.
  Fact 0.133: Jan Teige plays Edwina Reilly.
  Fact 0.134: Edwina Reilly is Bettina Reilly's twin sister.
  Fact 0.135: Edwina Reilly is just as snotty as Bettina Reilly.
  Fact 0.136: Irene Yah-Ling Sun plays Myrna Wong.
  Fact 0.137: Myrna Wong is a friend of Stella Johnson.
  Fact 0.138: Myrna Wong is a judo expert.
  Fact 0.139: Myrna Wong is assaulted by Kirby Baker.
  Fact 0.140: Myrna Wong led Kirby Baker on before he assaulted her.
  Fact 0.141: Myrna Wong destroys Kirby Baker and his office in retaliation.
  Fact 0.142: Myrna Wong has Kirby Baker sent to jail for assault.
  Fact 0.143: Tobias Anderson plays Barney Crunk, nicknamed 'Cranky'.
  Fact 0.144: Barney Crunk is the town's nosy mailman.
  Fact 0.145: Barney Crunk had the bad habit of peeking in everyone's mail.
  Fact 0.146: Barney Crunk is caught in flagrante delicto on film with Willamae Jones.
  Fact 0.147: Royce D. Applegate plays Dutch.
  Fact 0.148: Dutch is one of two kidnappers.
  Fact 0.149: Dutch abducts Skeeter Duggan at Flora Simpson-Reilly's orders.
  Fact 0.150: J.J. Barry plays Tex.
  Fact 0.151: Tex is the mastermind of the kidnap plot.
  Fact 0.152: The kidnap plot is meant to get Skeeter Duggan out of the way so Flora Simpson-Reilly and her friends could commit election fraud.
  Fact 0.153: Tex is arrested.
  Fact 0.154: Dutch is arrested.
  Fact 0.155: Dutch is Tex's accomplice.
  Fact 0.156: Tex and Dutch rat out Flora and her friends on their scheme.
  Fact 0.157: Pitt Herbert plays Henry Reilly.
  Fact 0.158: Henry Reilly is Flora's overly dominated husband.
  Fact 0.159: Henry Reilly is the grandfather of Edwina and Bettina.
  Fact 0.160: Henry Reilly is wealthy.
  Fact 0.161: Henry Reilly is very weak-willed.
  Fact 0.162: Henry Reilly is known for acquiescing to his wife's schemes.
  Fact 0.163: Arlen Stuart plays Bertha Harper.
  Fact 0.164: Bertha Harper is the wife of Otis Harper.
  Fact 0.165: Bertha Harper gets annoyed by Otis Harper's continual drinking.
  Fact 0.166: Bertha Harper is terrified when her house is destroyed by pink-painted elephants.

Retrieved items:
<item 1>
Graph query row:
cast_member: Pat Paulsen
character_role: starred
</item 1>

<item 2>
Harper Valley PTA (film)
He had a stampede of elephants , painted pink , stormed through his house , destroying it . Amzie Strickland as Shirley Thompson , a member of the PTA board who
tends to drink a lot of gin . Though she is allied with Flora and her friends , she is one of four PTA members ( Will , Skeeter and Holly Taylor are the others
) who are spared any retribution from Stella . She was seen as the most sympathetic of the board , as she tended to be more manipulated as opposed to being
manipulative . It was never revealed whether or not she retained her seat on the PTA , but it is presumed that she became more sympathetic towards Stella as she
was seen waving goodbye at Olive as she was being arrested . Bob Hastings as Skeeter Duggan , a member of the PTA board , and the town 's Notary public who is
sympathetic to Stella . He is kidnapped by Dutch and Tex at Flora Simpson Reilly 's orders . Audrey Christie as Flora Simpson - Reilly , The movie 's main
antagonist .
</item 2>

<item 3>
Harper Valley PTA (film)
Harper Valley PTA ( film ) - wikipedia Harper Valley PTA ( film ) Jump to : navigation , search Harper Valley PTA Theatrical release poster Directed by Richard
Bennett Produced by George Edwards Written by George Edwards Barry Schneider Starring Barbara Eden Nanette Fabray Ronny Cox Louis Nye Susan Swift Pat Paulsen
Music by Nelson Riddle Cinematography Willy Kurant Edited by Michael Economou Distributed by April Fools Productions Release date May 23 , 1978 ( 1978 - 05 - 23
) ( Midwest ) June 2 , 1978 ( 1978 - 06 - 02 ) ( Canada ) August 2 , 1978 ( 1978 - 08 - 02 ) ( Los Angeles ) January 12 , 1979 ( 1979 - 01 - 12 ) ( New York
City ) Running time 93 minutes Country United States Language English Budget $1 million Box office $25 million ( USA ) Harper Valley PTA is a 1978 American
comedy film inspired by the popular 1968 country song `` Harper Valley PTA '' written by Tom T. Hall and performed by country singer Jeannie C. Riley .
</item 3>

<item 4>
Harper Valley PTA (film)
Barbara Eden reprised her role as Stella Johnson for the series which lasted two seasons and a total of 30 episodes were produced . References ( edit ) Jump up
^ St. Petersburg Times Jump up ^ Milwaukee Sentinel : Elephant injures Nanette Fabray Jump up ^ Observer Reporter External links ( edit ) Harper Valley PTA on
IMDb Harper Valley PTA at the TCM Movie Database Harper Valley PTA at AllMovie Harper Valley PTA at the American Film Institute Catalog Retrieved from ``
https://en.wikipedia.org/w/index.php?title=Harper_Valley_PTA_(film)&oldid=818590841 '' Categories : 1978 films English - language films 1970s comedy films
American comedy films American films American independent films Feminist films Films about revenge Films adapted into television programs Films based on songs
Films set in Ohio Films shot in Ohio Films shot in Los Angeles 1978 soundtracks Film soundtracks Hidden categories : Articles with hAudio microformats Album
infoboxes lacking a cover Music infoboxes with deprecated parameters Talk Contents About Wikipedia Add links This page was last edited on 4 January 2018 , at 12
: 58 .
</item 4>

<item 5>
Graph query row:
cast_member: Ronny Cox
character_role: star
</item 5>

<item 6>
Graph query row:
cast_member: Nanette Fabray
character_role: starred
</item 6>

<item 7>
Harper Valley PTA (film)
He was arrested for indecent exposure after being set up by Stella . Ron Masak as Herbie Maddox , the head bartender at Kelly 's bar , and Stella 's ally
against Flora and her friends . He has a crush on Stella , and is one of her friends . Clint Howard as Corley , a young delivery person , who works for the
local florist shop , who brings Stella flowers from Will . Susan Swift as Dee Johnson , Stella 's 14 - year - old daughter , who is threatened with expulsion
from school if her mother does n't conform to the way the PTA wants her to behave . She has a hand in assisting her mother and Alice with one of their revenge
schemes . She loses her braces , gets a makeover , and ends up with a boyfriend , Carlyle . Pat Paulsen as Otis Harper , the descendant of the founding family
of Harper Valley . A PTA member who is allied with Flora and her friends . Has a tendency to drink all the time , which annoys his wife , Bertha . He had a
stampede of elephants , painted pink , stormed through his house , destroying it .
</item 7>

<item 8>
Harper Valley PTA (film)
The film grossed over $2 million its opening weekend , $5.8 million in its first three weeks and over $16 million after twelve weeks . The film opened in Los
Angeles on August 2 , 1978 and in New York City on January 12 , 1979 . Soundtrack ( edit ) Harper Valley PTA : Original Soundtrack Recording Soundtrack album by
various artists Released June 1978 Recorded Singleton Sound Studios Nashville , Tennessee Genre Soundtrack Length 29 : 50 Label Plantation Records Producer
Shelby S. Singleton , Jr . Singles from Harper Valley PTA `` Harper Valley PTA ( re-issue ) '' Released : July 1978 In June 1978 , a soundtrack album was
released on vinyl , cassette and 8 - track tape by Plantation Records . To promote the film 's release and its soundtrack , the title song by Jeannie C. Riley
was re-issued as a single . The album made its compact disc premiere through Varèse Sarabande on October 27 , 2017 .
</item 8>

<item 9>
Harper Valley PTA (film)
He is caught in flagrante delicto on film with Willamae Jones . Royce D. Applegate as Dutch , one of two kidnappers who abduct Skeeter Duggan at Flora Simpson -
Reilly 's orders . J.J. Barry as Tex , the mastermind of the kidnap plot to get Skeeter Duggan out of the way so Flora Simpson - Reilly and her friends could
commit election fraud . He and his accomplice Dutch are arrested and the two of them rat out Flora and her friends on their scheme . Pitt Herbert as Henry
Reilly , Flora 's overly dominated husband and grandfather to Edwina and Bettina . Henry is wealthy , but is very weak - willed and is known for acquiescing to
his wife 's schemes . Arlen Stuart as Bertha Harper , wife of Otis Harper , who gets annoyed by her husband 's continual drinking . She is terrified when her
house is destroyed by pink - painted elephants .
</item 9>

<item 10>
Graph query row:
cast_member: Barbara Eden
character_role: starred
</item 10>

<item 11>
Graph query row:
cast_member: Louis Nye
character_role: star
</item 11>

<item 12>
Harper Valley PTA (film)
Text is available under the Creative Commons Attribution - ShareAlike License ; additional terms may apply . By using this site , you agree to the Terms of Use
and Privacy Policy . Wikipedia ® is a registered trademark of the Wikimedia Foundation , Inc. , a non-profit organization . About Wikipedia
</item 12>

<item 13>
Harper Valley PTA (film)
The film starred Barbara Eden , Nanette Fabray , Ronny Cox , Louis Nye and Susan Swift , directed by Richard Bennett , and was mainly released to drive - in
theaters throughout the summer of 1978 . The film has a cult following in fans of the original song . The film 's promotional tagline is : `` The song was
scandalous . The movie is hilarious ! '' Contents ( hide ) 1 Plot 2 Cast and characters 3 Production and release 3.1 Filming 3.2 Release 4 Soundtrack 5
Television series 6 References 7 External links Plot ( edit ) Stella Johnson is a beautiful widowed single mother who lives in the town of Harper Valley , Ohio
. She sells cosmetics door - to - door for the fictitious AngelGlo Cosmetics and is not afraid to enjoy life . Her fourteen - year - old daughter , Dee , is a
student at Harper Valley Junior High School . After leaving school , Dee brings her mother a letter from the school 's Parent Teacher Association board , which
is led by the pompous and snobbish Flora Simpson - Reilly .
</item 13>

<item 14>
Harper Valley PTA (film)
She is terrified when her house is destroyed by pink - painted elephants . Production and release ( edit ) Filming ( edit ) Harper Valley PTA was filmed in
twenty - seven days from October 3 , 1977 to December 8 , 1977 . It was filmed on location for one week in the town of Lebanon , Ohio and then continued in Los
Angeles , California . The track meet scene was filmed at Simi Valley High School On October 31 , 1977 , production ceased temporarily when Nanette Fabray was
knocked to the ground by a spooked elephant while shooting a scene involving pink elephants in the Hancock Park neighborhood of Los Angeles , California ; she
suffered a severe concussion , bruises and back sprains and was taken to Cedars - Sinai Medical Center where her condition was reported as serious but stable .
Following her recovery , filming resumed on November 30 . Release ( edit ) Harper Valley PTA opened in six theaters in Lebanon , Cincinnati and Dayton , Ohio on
May 23 , 1978 in accordance with executive producer Phil Borack 's plan to release the film first in smaller markets , where regional success could encourage
exhibitors in bigger cities to book the film .
</item 14>

<item 15>
Graph query row:
cast_member: Susan Swift
character_role: star
</item 15>

<item 16>
Harper Valley PTA (film)
Side 1 `` Harper Valley PTA '' -- performed by Jeannie C. Riley `` Dee 's Visit '' ( instrumental ) -- performed by Nelson Riddle `` Mr. Harper '' -- performed
by Barbara Eden `` Alice 's Place '' ( instrumental ) -- performed by Nelson Riddle `` High School Confidential '' -- performed by Jerry Lee Lewis `` Willie Mae
'' ( instrumental ) -- performed by Nelson Riddle Side 2 `` Harper Valley PTA '' ( instrumental ) -- performed by Nelson Riddle `` Widow Jones '' -- performed
by Barbara Eden `` Twin Tune '' ( instrumental ) -- performed by Nelson Riddle `` Ballad of a Teenage Queen '' -- performed by Johnny Cash `` Ice Cream Disco ''
( instrumental ) -- performed by Nelson Riddle `` Whatever Happened to Charlie Brown '' -- performed by Rita Remington & Carol Channing `` Ending ( Reprise :
Harper Valley PTA ) '' ( instrumental ) -- performed by Nelson Riddle Television series ( edit ) Main article : Harper Valley PTA ( TV series ) In 1981 , Harper
Valley PTA was made into a television sitcom ( created by Sherwood Schwartz ) which aired on NBC from January 1981 to May 1982 .
</item 16>
=== END ITEM id=094e6fe4a99e7f13 ===

=== ITEM id=70479f8f1ef8f9a7 ===
Gold sentences and facts:
Sentence 0: For thousands of years , seawater pearls were retrieved by divers in the Indian Ocean in areas such as the Persian Gulf , the Red Sea and the Gulf of Mannar . Evidence also suggest a prehistoric origin to pearl diving in these regions . Starting in the Han Dynasty ( 206 BC -- 220 AD ) , the Chinese hunted extensively for seawater pearls in the South China Sea . In the 14th - century Arabian Sea , the traveller Ibn Battuta provided the earliest known description of pearl diving by means of attaching a cord to the diver 's waist .
  Fact 0.0: For thousands of years, seawater pearls were retrieved by divers in the Indian Ocean.
  Fact 0.1: Divers retrieved seawater pearls in areas such as the Persian Gulf.
  Fact 0.2: Divers retrieved seawater pearls in areas such as the Red Sea.
  Fact 0.3: Divers retrieved seawater pearls in areas such as the Gulf of Mannar.
  Fact 0.4: Evidence also suggests a prehistoric origin to pearl diving in these regions.
  Fact 0.5: Starting in the Han Dynasty (206 BC to 220 AD), the Chinese hunted extensively for seawater pearls in the South China Sea.
  Fact 0.6: In the 14th-century Arabian Sea, the traveller Ibn Battuta provided the earliest known description of pearl diving.
  Fact 0.7: Ibn Battuta's description of pearl diving involved attaching a cord to the diver's waist.

Retrieved items:
<item 1>
Pearl
Australia also has one of the world 's last remaining fleets of pearl diving ships . Australian pearl divers dive for south sea pearl oysters to be used in the
cultured south sea pearl industry . The catch of pearl oysters is similar to the numbers of oysters taken during the natural pearl days . Hence significant
numbers of natural pearls are still found in the Australian Indian Ocean waters from wild oysters . X-ray examination is required to positively verify natural
pearls found today . Types of cultured pearls ( edit ) A blister pearl , a half - sphere , formed flush against the shell of the pearl oyster . Keshi pearls ,
although they often occur by chance , are not considered natural . They are a byproduct of the culturing process , and hence do not happen without human
intervention . They are quite small , typically only a few millimeters . Keshi pearls are produced by many different types of marine mollusks and freshwater
mussels in China . Keshi pearls are actually a mistake in the cultured pearl seeding process . In seeding the cultured pearl , a piece of mantle muscle from a
sacrificed oyster is placed with a bead of mother of pearl within the oyster .
</item 1>

<item 2>
Pearl
Freshwater and saltwater pearls ( edit ) Freshwater and saltwater pearls may sometimes look quite similar , but they come from different sources . Freshwater
pearls form in various species of freshwater mussels , family Unionidae , which live in lakes , rivers , ponds and other bodies of fresh water . These
freshwater pearl mussels occur not only in hotter climates , but also in colder more temperate areas such as Scotland ( where they are protected under law ) .
Most freshwater cultured pearls sold today come from China . Saltwater pearls grow within pearl oysters , family Pteriidae , which live in oceans . Saltwater
pearl oysters are usually cultivated in protected lagoons or volcanic atolls . Creation ( edit ) Diagram comparing a cross-section of a cultured pearl , upper ,
with a natural pearl , lower Pearls are formed inside the shell of certain mollusks as a defense mechanism against a potentially threatening irritant such as a
parasite inside the shell , or an attack from outside that injures the mantle tissue . The mollusk creates a pearl sac to seal off the irritation .
</item 2>

<item 3>
Pearl
While white , and more recently black , saltwater pearls are by far the most popular , other color tints can be found on pearls from the oceans . Pink , blue ,
champagne , green , black and even purple saltwater pearls can be encountered , but to collect enough of these rare colors to form a complete string of the same
size and same shade can take years . Religious references ( edit ) This section needs additional citations for verification . Please help improve this article
by adding citations to reliable sources . Unsourced material may be challenged and removed . ( March 2018 ) ( Learn how and when to remove this template message
) Hindu scriptures ( edit ) The Hindu tradition describes the sacred Nine Pearls which were first documented in the Garuda Purana , one of the books of the
Hindu mythology . Ayurveda contains references to pearl powder as a stimulant of digestion and to treat mental ailments . According to Marco Polo , the kings of
Malabar wore a necklace of 104 rubies and pearls which was given from one generation of kings to the next . The reason was that every king had to say 104
prayers every morning and every evening .
</item 3>

<item 4>
Pearl
Prized for their exquisitely beautiful ' orient ' or lustre , South Sea pearls are now farmed in various parts of the world where the Pinctada maxima oysters
can be found , with the finest South Sea pearls being produced by Paspaley along the remote coastline of North - Western Australia . White and silver colored
South Sea pearls tend to come from the Broome area of Australia , while golden colored ones are more prevalent in the Philippines and Indonesia . A farm in the
Gulf of California , Mexico , is culturing pearls from the black lipped Pinctada mazatlanica oysters and the rainbow lipped Pteria sterna oysters . Also called
Concha Nácar , the pearls from these rainbow lipped oysters fluoresce red under ultraviolet light . From other species ( edit ) A shell of the Indian volute ,
Melo melo , surrounded by a number of pearls from this species Conch pearl pendant Biologically speaking , under the right set of circumstances , almost any
shelled mollusk can produce some kind of pearl . However , most of these molluskan pearls have no luster or iridescence .
</item 4>

<item 5>
Pearl
Pearls of abalone , or pāua , are mabe pearls , or blister pearls , unique to New Zealand waters and are commonly referred to as ' blue pearls ' . They are
admired for their incredible luster and naturally bright vibrant colors that are often compared to opal . Another example is the conch pearl ( sometimes
referred to simply as the ' pink pearl ' ) , which is found very rarely growing between the mantle and the shell of the queen conch or pink conch , Strombus
gigas , a large sea snail or marine gastropod from the Caribbean Sea . These pearls , which are often pink in color , are a by - product of the conch fishing
industry , and the best of them display a shimmering optical effect related to chatoyance known as ' flame structure ' . Somewhat similar gastropod pearls ,
this time more orange in hue , are ( again very rarely ) found in the horse conch Pleuroploca gigantea . The second largest pearl known was found in the
Philippines in 1934 and is known as the Pearl of Lao Tzu .
</item 5>

<item 6>
Pearl
It also records that eight varieties of pearls accompanied Prince Vijaya 's embassy to the Pandyan king as well as king Devanampiya Tissa 's embassy to Emperor
Ashoka . Pliny the Elder ( 23 -- 79AD ) praised the pearl fishery of the Gulf as most productive in the world . For thousands of years , seawater pearls were
retrieved by divers in the Indian Ocean in areas such as the Persian Gulf , the Red Sea and the Gulf of Mannar . Evidence also suggest a prehistoric origin to
pearl diving in these regions . Starting in the Han Dynasty ( 206 BC -- 220 AD ) , the Chinese hunted extensively for seawater pearls in the South China Sea .
In the 14th - century Arabian Sea , the traveller Ibn Battuta provided the earliest known description of pearl diving by means of attaching a cord to the diver
's waist . Catching of pearls , Bern Physiologus ( 9th century ) When Spanish conquistadors arrived in the Western Hemisphere , they discovered that around the
islands of Cubagua and Margarita , some 200 km north of the Venezuelan coast , was an extensive pearl bed ( a bed of pearl oysters ) .
</item 6>

<item 7>
Pearl
They are , for the most part , freshwater pearls from mussels . Pearling was banned in the U.K. in 1998 due to the endangered status of river mussels .
Discovery and publicity about the sale for a substantial sum of the Abernethy pearl in the River Tay had resulted in heavy exploitation of mussel colonies
during the 1970s and 80s by weekend warriors . When it was permitted it was carried on mainly by Scottish Travellers who found pearls varied from river to river
with the River Oykel in the Highlands being noted for the finest rose - pink pearls . There are two firms in Scotland that are licensed to sell pre-1998
freshwater pearls . Pearl farming ( edit ) A pearl being extracted from an akoya pearl oyster . See also : Oyster farming Today , the cultured pearls on the
market can be divided into two categories . The first category covers the beaded cultured pearls , including Akoya , South Sea and Tahiti . These pearls are
gonad grown , and usually one pearl is grown at a time . This limits the number of pearls at a harvest period .
</item 7>

<item 8>
Pearl
1 British Isles 7.2 Pearl farming 8 Timeline of pearl production 9 Freshwater pearl farming 10 Momme weight 11 In jewelry 11.1 Gallery 11.2 Shapes 11.3 Lengths
of pearl necklaces 11.4 Colors 12 Religious references 12.1 Hindu scriptures 12.2 Hebrew scriptures 12.3 New Testament scriptures 12.4 Islamic scriptures 12.5
Additional references 13 See also 14 References 15 External links Etymology ( edit ) The English word pearl comes from the French perle , originally from the
Latin perna meaning leg , after the ham - or mutton leg - shaped bivalve . Definition ( edit ) A black pearl and a shell of the black - lipped pearl oyster .
The iridescent colors originate from nacre layers . All shelled mollusks can , by natural processes , produce some kind of `` pearl '' when an irritating
microscopic object becomes trapped within its mantle folds , but the great majority of these `` pearls '' are not valued as gemstones .
</item 8>

<item 9>
Pearl
It is a naturally occurring , non-nacreous , calcareous concretion ( pearl ) from a giant clam . Because it did not grow in a pearl oyster it is not pearly ;
instead the surface is glossy like porcelain . Other pearls from giant clams are known to exist , but this is a particularly large one weighing 14 lb ( 6.4 kg )
. The largest known pearl ( also from a giant clam ) was found in the Philippines as well by a fisherman from Puerto Princesa , Palawan Island . The enormous
pearl is 30 cm wide ( 1 ft ) , 67 cm long ( 2.2 ft ) and weighs 75 lb ( 34 kg ) . History ( edit ) Pearl hunting ( edit ) Main article : Pearl hunting A 14th -
century piece of clothing used by Kuwaiti divers searching for pearls in the Persian Gulf The ancient chronicle Mahavamsa mentions of the thriving pearl
industry in the port of Oruwella in the Gulf of Mannar in Sri Lanka .
</item 9>

<item 10>
Pearl
Nacreous pearls , the best - known and most commercially significant , are primarily produced by two groups of molluskan bivalves or clams . A nacreous pearl is
made from layers of nacre , by the same living process as is used in the secretion of the mother of pearl which lines the shell . Natural ( or wild ) pearls ,
formed without human intervention , are very rare . Many hundreds of pearl oysters or mussels must be gathered and opened , and thus killed , to find even one
wild pearl ; for many centuries , this was the only way pearls were obtained , and why pearls fetched such extraordinary prices in the past . Cultured pearls
are formed in pearl farms , using human intervention as well as natural processes . One family of nacreous pearl bivalves -- the pearl oyster -- lives in the
sea , while the other -- a very different group of bivalves -- lives in freshwater ; these are the river mussels such as the freshwater pearl mussel . Saltwater
pearls can grow in several species of marine pearl oysters in the family Pteriidae .
</item 10>
=== END ITEM id=70479f8f1ef8f9a7 ===
