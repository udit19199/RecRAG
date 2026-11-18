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

=== ITEM id=48bafe72d8b8e4e6 ===
Gold sentences and facts:
Sentence 0: `` It Ai n't Me '' is a song by Norwegian DJ Kygo and American singer Selena Gomez . It was released by Interscope Records , Sony and Ultra on 17 February 2017 as the lead single from Kygo 's first EP , Stargazing . The song was written by Kygo , Gomez , Andrew Watt , Brian Lee and Ali Tamposi . It was produced by Kygo , Watt , Ben Rice and Louis Bell . A dance - pop , electropop and tropical house song , `` It Ai n't Me '' comprises an acoustic guitar line , and a build - drop arrangement in its chorus featuring pulsing piano notes , bass , synthesizers , finger - snap claps and pan flute melodies . Gomez sings the track in a husky tone , while in the chorus her vocals are reduced to recurring syllables . The lyrics are nostalgic and narrate a past relationship ruined by alcoholism and partying too often .
  Fact 0.0: 'It Ai n't Me' is a song by Norwegian DJ Kygo and American singer Selena Gomez.
  Fact 0.1: 'It Ai n't Me' was released by Interscope Records, Sony and Ultra on 17 February 2017.
  Fact 0.2: 'It Ai n't Me' was the lead single from Kygo's first EP, Stargazing.
  Fact 0.3: 'It Ai n't Me' was written by Kygo, Selena Gomez, Andrew Watt, Brian Lee and Ali Tamposi.
  Fact 0.4: 'It Ai n't Me' was produced by Kygo, Andrew Watt, Ben Rice and Louis Bell.
  Fact 0.5: 'It Ai n't Me' is a dance-pop, electropop and tropical house song.
  Fact 0.6: 'It Ai n't Me' comprises an acoustic guitar line.
  Fact 0.7: 'It Ai n't Me' has a build-drop arrangement in its chorus.
  Fact 0.8: The chorus of 'It Ai n't Me' features pulsing piano notes, bass, synthesizers, finger-snap claps and pan flute melodies.
  Fact 0.9: Selena Gomez sings the track in a husky tone.
  Fact 0.10: In the chorus of 'It Ai n't Me', Selena Gomez's vocals are reduced to recurring syllables.
  Fact 0.11: The lyrics of 'It Ai n't Me' are nostalgic.
  Fact 0.12: The lyrics of 'It Ai n't Me' narrate a past relationship ruined by alcoholism and partying too often.

Retrieved items:
<item 1>
It Ain't Me
Enter It Ai n't Me in the search field and then press Enter . Jump up ^ `` American single certifications -- Kygo & Selena Gomez -- It Ai n't Me '' . Recording
Industry Association of America . Retrieved 7 June 2017 . If necessary , click Advanced , then click Format , then select Single , then click SEARCH Jump up ^
`` Kygo & Selena Gomez - It Ai n't Me RadioDate '' . Radioairplay.fm. 16 February 2017 . Retrieved 9 September 2017 . Jump up ^ ( 1 ) Jump up ^ `` BBC - Radio 1
- Playlist '' . BBC Radio 1 . 17 March 2017 . Archived from the original on 18 March 2017 . Retrieved 21 June 2017 . Jump up ^ `` Music Meeting for March 28 ''
. Play IGA . Archived from the original on 28 March 2017 . External links ( edit ) Lyrics of this song at MetroLyrics Kygo Discography Studio albums Cloud Nine
( 2016 ) Kids in Love ( 2017 ) EPs Stargazing ( 2017
</item 1>

<item 2>
It Ain't Me
For the Bob Dylan song , see It Ai n't Me , Babe . `` It Ai n't Me '' Single by Kygo and Selena Gomez from the EP Stargazing Released 17 February 2017 Format
Digital download Recorded 2016 Genre Dance - pop electropop tropical house Length 3 : 40 Label Interscope Sony Ultra Songwriter ( s ) Kyrre Gørvell - Dahll
Selena Gomez Andrew Wotman Brian Lee Ali Tamposi Producer ( s ) Kygo Andrew Watt Ben Rice Louis Bell Kygo singles chronology `` Carry Me '' ( 2016 ) `` It Ai
n't Me '' ( 2017 ) `` First Time '' ( 2017 ) Selena Gomez singles chronology `` Trust Nobody '' ( 2016 ) Trust Nobody 2016 `` It Ai n't Me '' ( 2017 ) It Ai n't
Me2017 `` Bad Liar '' ( 2017 ) Bad Liar 2017 Music video `` It Ai n't Me '' on YouTube `` It Ai n't Me '' is a song by Norwegian DJ Kygo and American singer
Selena Gomez .
</item 2>

<item 3>
It Ain't Me
It also reached the top 10 in Italy , Spain , the United Kingdom and the United States . Contents ( hide ) 1 Writing and recording 2 Music and lyrics 3 Release
4 Critical reception 5 Chart performance 6 Music video 7 Usage in media 8 Track listing 9 Credits and personnel 10 Charts 10.1 Weekly charts 10.2 Year - end
charts 11 Certifications 12 Release history 13 References 14 External links Writing and Recording ( edit ) The lyrics of `` It Ai n't Me '' were written by
Andrew Watt , Brian Lee and Ali Tamposi . With Watt previously working with Kygo , the DJ 's manager Myles Shear contacted Watt for a session with Kygo which
Watt also facilitated for Lee and Tamposi . During their session for `` It Ai n't Me '' , they initially found the track very poor , but were trying to salvage
it . Kygo and Shear suggested that they continue working on the song .
</item 3>

<item 4>
It Ain't Me
The same synthesizers from the verses are repeated with a piano in the background . Gomez 's vocals in the drop are reduced to syllables from her verses such as
a recurring `` whiskey neat '' and are accompanied by percussion instruments . `` It Ai n't Me '' is a breakup song with a theme of nostalgia . The lyrics are
about standing up for yourself in the face of a toxic relationship . Gomez narrates the regret of a previous relationship ruined by her former lover 's habits
of drinking and partying too often . The lyrics specifically address alcohol abuse , a type of alcoholism . Tamposi explained that `` It Ai n't Me '' is about a
woman finding the strength to walk away from a toxic relationship despite the pressure she receives from society to stand by her man . Although the lyrics are
sombre , the song has a distinctly uplifting and empowering sound . Release ( edit ) In early February 2017 , `` It Ai n't Me '' was registered on the American
Society of Composers , Authors and Publishers ( ASCAP ) database . Gomez first teased the single with two Instagram stories on 3 February 2017 .
</item 4>

<item 5>
It Ain't Me
It Ai n't Me - wikipedia It Ai n't Me Jump to : navigation , search This article is about the Kygo and Selena Gomez song . For the Bob Dylan song , see It Ai
n't Me , Babe .
</item 5>

<item 6>
It Ain't Me
It was released by Interscope Records , Sony and Ultra on 17 February 2017 as the lead single from Kygo 's first EP , Stargazing . The song was written by Kygo
, Gomez , Andrew Watt , Brian Lee and Ali Tamposi . It was produced by Kygo , Watt , Ben Rice and Louis Bell . A dance - pop , electropop and tropical house
song , `` It Ai n't Me '' comprises an acoustic guitar line , and a build - drop arrangement in its chorus featuring pulsing piano notes , bass , synthesizers ,
finger - snap claps and pan flute melodies . Gomez sings the track in a husky tone , while in the chorus her vocals are reduced to recurring syllables . The
lyrics are nostalgic and narrate a past relationship ruined by alcoholism and partying too often . In addition to reaching number one in Croatia , Lebanon , and
Norway , the song attained a top five peak in Australia , Austria , Belgium , Canada , the Czech Republic , Denmark , Finland , France , Germany , Greece ,
Hungary , Ireland , the Netherlands , New Zealand , Poland , Portugal , Slovakia , Sweden and Switzerland .
</item 6>

<item 7>
It Ain't Me
Usage in Media ( edit ) The song is featured in tenth episode of the television series The Bold Type . Track listing ( edit ) Digital download `` It Ai n't Me
'' -- 3 : 40 Digital download -- Tiësto 's AFTR : HRS Remix `` It Ai n't Me '' ( Tiësto 's AFTR : HRS Remix ) -- 3 : 12 Credits and personnel ( edit ) Credits
adapted from Qobuz . Kygo -- composer , producer Selena Gomez -- composer , vocals Andrew Watt -- guitar , producer , vocal producer , composer , songwriter
Brian Lee -- composer , songwriter Ali Tamposi -- composer , songwriter Tom Coyne -- mastering engineer Serban Ghenea -- mixing engineer Ben Rice -- vocal
producer Louis Bell -- vocal producer Charts ( edit ) Weekly charts ( edit ) Chart ( 2017 ) Peak position Australia ( ARIA ) Austria ( Ö3 Austria Top 40 )
Belgium ( Ultratop 50 Flanders ) Belgium ( Ultratop 50 Wallonia ) Canada ( Canadian Hot 100 ) Canada CHR / Top 40 ( Billboard ) Canada Hot AC ( Billboard ) 6
Colombia ( Monitor Latino Anglo ) Croatia ( HRT ) Czech Republic (
</item 7>

<item 8>
It Ain't Me
The song carries a calm tone and a drop common in tropical house music , but is complimented by an equal measure of electronic and pop rock elements . The track
starts with a folk - like acoustic guitar riff . It is followed immediately by Gomez singing the first verse in a husky tone . As the song develops , a steady
bass and synthesizers are heard . The production is subtle and varied with finger snaps sometimes being louder than the kick drum , and the guitar riff and
piano notes alternating in and out . In the chorus , Gomez is accompanied by background vocals that pertain to the sound of a choir . `` It Ai n't Me '' A 21 -
second sound sample from `` It Ai n't Me '' featuring the chorus to the start of the drop which Gomez sings accompanied by backing vocals , piano notes , a
guitar riff , and vocal effects . Problems playing this file ? See media help . The chorus has a new melodic arrangement and a crescendo . It is ensued by chord
- buoyed builds and drops comprising finger snaps and pan flute melodies . The same synthesizers from the verses are repeated with a piano in the background .
</item 8>

<item 9>
It Ain't Me
by Brian Lee ( songwriter ) Songs about alcohol Tropical house songs Interscope Records singles Hidden categories : CS1 Norwegian - language sources ( no )
Pages containing links to subscription - only content CS1 French - language sources ( fr ) CS1 Spanish - language sources ( es ) CS1 Japanese - language sources
( ja ) CS1 Turkish - language sources ( tr ) CS1 German - language sources ( de ) CS1 Italian - language sources ( it ) CS1 Polish - language sources ( pl ) CS1
Swedish - language sources ( sv ) CS1 Portuguese - language sources ( pt ) All articles lacking reliable references Articles lacking reliable references from
January 2017 Use dmy dates from February 2017 Music infoboxes with deprecated parameters Articles with hAudio microformats Pages using div col with deprecated
parameters Singlechart usages for Australia Singlechart usages for Austria Singlechart usages for Flanders Singlechart usages for Wallonia Singlechart usages
for Billboardcanadachrtop40 Singlechart called without song Singlechart usages for Billboardcanadahotac Singlechart usages for Czech Republic Singlechart called
without artist Singlechart usages for Czechdigital Singlechart usages for Denmark Singlechart usages for Finland Singlechart usages
</item 9>

<item 10>
It Ain't Me
000 United States ( RIAA ) Platinum 1,000,000 sales figures based on certification alone shipments figures based on certification alone sales + streaming
figures based on certification alone Release history ( edit ) Region Date Version Format Label Ref . Italy 16 February 2017 Original Contemporary hit radio Sony
Various 17 February 2017 Single Digital download Sony Ultra United States 28 February 2017 Original Mainstream radio Ultra 7 March 2017 Rhythmic radio Sony
Ultra United Kingdom 17 March 2017 Contemporary hit radio United States 28 March 2017 Hot adult contemporary Various 7 April 2017 Tiësto 's AFTR : HRS Remix
Digital download References ( edit ) Jump up ^ https://www.pandora.com/artist/kygo-and-selena-gomez/it-aint-me-single/it-aint-me/TR9c9kdKhgc9Xcm ^ Jump up to :
Bein , Kat ( 23 February 2017 ) . `` Selena Gomez Is ' an Old Soul With a Beautiful Baby Face , ' Says ' It Ai n't Me ' Songwriter '' . Billboard . Retrieved 24
February 2016 .
</item 10>
=== END ITEM id=48bafe72d8b8e4e6 ===

=== ITEM id=edbc90c530461337 ===
Gold sentences and facts:
Sentence 0: Diary of a Wimpy Kid : The Getaway Author Jeff Kinney Country United States Language English Series Diary of a Wimpy Kid Genre Children 's novel Graphic Novel Publisher Amulet Books ( US ) Puffin Books ( UK ) Publication date November 7 , 2017 ( Worldwide ) Media type Print ( hardcover ) Pages 217 story pages ( 224 altogether ) ISBN 978 - 1 - 4197 - 2545 - 6 Preceded by Double Down
  Fact 0.0: The title is Diary of a Wimpy Kid: The Getaway.
  Fact 0.1: The author is Jeff Kinney.
  Fact 0.2: The country is United States.
  Fact 0.3: The language is English.
  Fact 0.4: The series is Diary of a Wimpy Kid.
  Fact 0.5: The genres are children's novel and graphic novel.
  Fact 0.6: The publisher is Amulet Books (US).
  Fact 0.7: The publisher is Puffin Books (UK).
  Fact 0.8: The publication date is November 7, 2017 (worldwide).
  Fact 0.9: The media type is print (hardcover).
  Fact 0.10: The book has 217 story pages, 224 altogether.
  Fact 0.11: The ISBN is 978-1-4197-2545-6.
  Fact 0.12: It is preceded by Double Down.

Retrieved items:
<item 1>
Diary of a Wimpy Kid: The Getaway
They each wear some of the clothes in there ; they first head to the clubhouse , and then the putting green for golf . However , they are spotted by a security
guard , who escorts them out of the green . A family had reported them for wearing their missing clothes . They attempted a failed getaway with a golf cart ,
and are ultimately sent back to the hotel to return the clothes . They were also ejected from the hotel for theft , so they were forced to stay overnight at the
airport hotel . The next day , Susan refuses to leave the resort to leave on a high note . So they sneak in . They skip the pool and then find Rodrick 's
girlfriend with a man named Rodrigo . They go back to the beach , and scavenge for food by distracting others and then eating their food . They go to take one
more photo , but are spotted by the family who they took clothes from . The Heffleys try to escape the resort . Greg heads for the beach and windsurfs away . He
gets stuck in the dolphin swimming area , but is chased back to shore by security .
</item 1>

<item 2>
Diary of a Wimpy Kid: The Getaway
They first try the banana boat , but this is ruined when a kid using a water trampoline lands on and bursts their float . They get hungry , so they go to the
golf clubhouse which turns them down as they are not following the dress code . They go to the swim - up bar , and learn of a monkey which has been returning
there for drinks every day . They are chased away when Manny finds the TV remote , changes the channel to his favorite cartoons , and throws the remote in the
water . Greg goes to the arcade , and the family regroup later to find Rodrick . The same girl he helped earlier is now his girlfriend . In the evening , they
go to the beach bonfire and go to see turtles hatching . Just as they are prepared to leave , they begin to hatch , but they lose their way to the sea as people
are using the flashes on their phones to take photos . The next day , Susan goes to the spa and Frank goes to the gym . Rodrick then goes back to the teen zone
.
</item 2>

<item 3>
The Sea of Monsters
In March 2012 , Nathan Fillion joined the cast as Hermes and Anthony Head , replacing Pierce Brosnan , as Chiron . In April 2012 , Paloma Kwiatkowski was cast
as Thalia and The Hunger Games star Leven Rambin has been cast as Clarisse La Rue . It was announced on May 8 , 2012 that the film had moved up to March 15 ,
2013 . But later it was postponed for an August 16 , 2013 release . Fox decided on a final release date for August 7 , 2013 . Logan Lerman returned as Percy
Jackson in the movie . Jake Abel also returned to the role of Luke Castellan . On January 2012 The Secret Circle star Grey Damon was cast as Chris Rodriguez .
According to Alexandra Daddario , the film would be more similar to the book than the first film . Graphic novel ( edit ) The Sea of Monsters has been published
as a graphic novel . Sequel ( edit ) Main article : Percy Jackson and the Titan 's Curse The novel was followed by The Titan 's Curse , released on May 1 , 2007
.
</item 3>

<item 4>
Diary of a Wimpy Kid: The Getaway
Waterstones.com . Retrieved 10 December 2017 . ( hide ) The Diary of a Wimpy Kid series by Jeff Kinney Novels Diary of a Wimpy Kid Rodrick Rules The Last Straw
Dog Days The Ugly Truth Cabin Fever The Third Wheel Hard Luck The Long Haul Old School Double Down The Getaway Characters Greg Heffley Film series Diary of a
Wimpy Kid Rodrick Rules Dog Days The Long Haul Retrieved from `` https://en.wikipedia.org/w/index.
</item 4>

<item 5>
Diary of a Wimpy Kid: The Getaway
The book was unveiled during the 2017 Diary of a Wimpy Kid Virtually Live Event which was live streamed via YouTube as part of the 10th anniversary of the first
book . The book was published and released on November 7 , 2017 . Contents ( hide ) 1 Plot 2 Release date 3 Reception 4 References Plot ( edit ) After many
frustrations , Greg Heffley 's parents decide to skip Christmas and travel by plane to Isla De Corales , the hotel they stayed at while on their honeymoon much
to Greg 's dismay . On Christmas Eve they drive to the airport , but are delayed by a series of mishaps . They encounter holiday traffic but arrive at the
airport . As they fail to unload in time at the drop off zone , Frank and Greg have to return later with the luggage . Greg steps in a slush puddle and they end
up missing the shuttle . In the airport , the family 's suitcases are too heavy , so they stuff some clothes into their carry on baggage .
</item 5>

<item 6>
Diary of a Wimpy Kid: The Getaway
He is even excited by the activities the resort offer which he sees on the bus . However , Frank and Susan are disappointed that the resort has since changed ,
and it is now split with a family side ( the Mild Side ) and a couple side ( the Wild Side ) . The family also discovers that they got the wrong luggage that is
locked , and their luggage was sent back home . They only have the clothes they put in their carry on luggage . They attempt to buy some new clothes , but Frank
wo n't buy them as they are too expensive . The family goes to explore the resort and end up at the pool . The Director of Fun shows up , however , and gets
everybody moving . Greg is forced to take part in two games . After this , Greg sees an iguana by the pool and is worried , so the family goes to get some lunch
. Greg and his family try to eat outside ; however , too many birds and bugs keep ruining their meals . Frank contracts a stomach bug from his water , so they
head back to the suite . The next day , Greg uses the shower which is outside .
</item 6>

<item 7>
Diary of a Wimpy Kid: The Getaway
Release date ( edit ) The Getaway is the 12th book in the series , and was released everywhere except Canada and China on November 7 , 2017 . Reception ( edit )
A week since its release , it has received 4.5 / 5 stars on Waterstones with 786 reviews . References ( edit ) Jump up ^ Wimpy Kid ( 27 April 2017 ) . `` Wimpy
Kid LIVE Special 10th Anniversary Presentation and Cover Reveal of Book 12 ! '' . YouTube . Retrieved 10 December 2017 . Jump up ^ `` Cover Of New Diary Of A
Wimpy Kid Book Revealed By Jeff Kinney To Fans Worldwide At Live streamed Carnegie Hall Event Hosted By ABRAMS '' . Tristateupdate.com . Retrieved 10 December
2017 . Jump up ^ `` Diary of a Wimpy Kid : The Getaway by Jeff Kinney -- Waterstones '' . Waterstones.com . Retrieved 10 December 2017 .
</item 7>

<item 8>
Diary of a Wimpy Kid: The Getaway
Because the pool workers could n't find the jellyfish , they have to drain the pool . They escape in the commotion , and notice Rodrick is n't there . He had
been left in the sea where they went scuba diving and had extreme sunburn . They return to the suite and decide to spend their last few days outdoors . The next
day , Greg goes to the tennis court in hopes of learning how to play tennis . Unfortunately , it was just a cage full of children with a ball machine and tennis
balls . It turns into a free - for - all , which stops suddenly when a kid recognizes Greg as the family who brought a box jellyfish to the poolside . He tries
to tell them that it was Manny 's fault , but they throw tennis balls at him . He quickly escapes the cage and goes back to the suite with his family . They
were out of things to do and want to leave the resort early ; however , Frank refuses to leave until he gets a decent meal . Manny discovers that the suitcase
they got on mistake fell and broke and also has the clothes they need to get in the clubhouse .
</item 8>

<item 9>
Diary of a Wimpy Kid: The Getaway
Please discuss this issue on the article 's talk page . ( December 2017 ) This article needs more links to other articles to help integrate it into the
encyclopedia . Please help improve this article by adding links that are relevant to the context within the existing text . ( December 2017 ) ( Learn how and
when to remove this template message ) ( Learn how and when to remove this template message ) Diary of a Wimpy Kid : The Getaway Author Jeff Kinney Country
United States Language English Series Diary of a Wimpy Kid Genre Children 's novel Graphic Novel Publisher Amulet Books ( US ) Puffin Books ( UK ) Publication
date November 7 , 2017 ( Worldwide ) Media type Print ( hardcover ) Pages 217 story pages ( 224 altogether ) ISBN 978 - 1 - 4197 - 2545 - 6 Preceded by Double
Down Diary of a Wimpy Kid : The Getaway is the twelfth book in the Diary of a Wimpy Kid series by Jeff Kinney .
</item 9>

<item 10>
Diary of a Wimpy Kid: The Getaway
Diary of a Wimpy Kid : the Getaway - wikipedia Diary of a Wimpy Kid : the Getaway Jump to : navigation , search ( hide ) This article has multiple issues .
Please help improve it or discuss these issues on the talk page . ( Learn how and when to remove these template messages ) This article consists almost entirely
of a plot summary . It should be expanded to provide more balanced coverage that includes real - world context . Please edit the article to focus on discussing
the work rather than merely reiterating the plot . ( August 2016 ) ( Learn how and when to remove this template message ) This article needs additional
citations for verification . Please help improve this article by adding citations to reliable sources . Unsourced material may be challenged and removed . (
December 2017 ) ( Learn how and when to remove this template message ) This article 's lead section does not adequately summarize key points of its contents .
Please consider expanding the lead to provide an accessible overview of all important aspects of the article . Please discuss this issue on the article 's talk
page .
</item 10>
=== END ITEM id=edbc90c530461337 ===

=== ITEM id=c90252338c16c857 ===
Gold sentences and facts:
Sentence 0: Wizards of Waverly Place Season 4 episodes No . overall No. in season Title Directed by Written by Original air date Prod . code U.S. viewers ( millions ) 80 `` Alex Tells the World '' Victor Gonzalez Gigi McCreery & Perry Rein November 12 , 2010 ( 2010 - 11 - 12 ) 401 4.1 The Russos and Mason finally make it back home , but their wizard powers are still gone . While the family wants to stay quiet , Alex believes that they should take action , so she reveals that they are wizards to reporters and tells them that a group of wizards are being held captive by the government . As she urges them to spread the word , Professor Crumbs appears and tells her that everything that had happened was a test that she failed miserably and orders her to Wizard Court for exposing wizardry to the world . Justin is also ordered to attend because he exposed magic to Agent Lamwood , who was a creation from Professor Crumbs ' mind . The judges rule Alex and Justin guilty , and Professor Crumbs sentences them both to be demoted to level one in the Family Wizard Competition , making them behind Max by two levels . Meanwhile , Max sets up traps in the Sub Station for the `` government '' , but ends up falling for one of them . Special guest star : Gregg Sulkin as Mason Guest star : Ian Abercrombie as Professor Crumbs 81 `` Alex Gives Up '' Victor Gonzalez Todd J. Greenwald November 27 , 2010 ( 2010 - 11 - 27 ) 402 N / A Alex decides to give up the wizard competition to spend more time with Mason and thinks that it does not matter since Max is all the way in the lead , but Chancellor Tootietootie tells Alex that non-wizards and werewolves can not date , because the werewolf always eats the human . Max 's picture on the front of Future Wizards magazine attracts the daughter of a family of Cucuys ( mythical Latino monster ) and they invite him to a dinner party on their yacht . Alex comes up with a scheme to invite Chancellor Tootietootie to the Cucuy party in hopes of getting him to appeal the law against werewolves and mortals dating . Lisa Cucuy finds out that Mason is a werewolf and decides that she wants him instead of Max . Alex gets jealous when Lisa flirts with Mason , so Alex says embarrassing things about Mason to everyone on the yacht so that he will be mad enough to turn into a werewolf . Carlos and Julie Cucuy jump off the boat as Cucuys are scared of werewolves . Jerry jumps off after them . Alex tells Lisa that she can get off the boat the easy way or her way . Once Lisa jumps off the boat , Chancellor Tootietootie says that their case will not be appealed as by Mason turning into a werewolf proved his point that they are dangerous . In the end , Alex and Mason have to break up to be friends and Harper finishes writing her first novel based on the werewolf - wizard relationship . Jerry tries to get Harper to help him out of the water but accidentally grabs her just finished novel , which causes him to fall back into the water . Meanwhile , Justin tutors a class of delinquent wizards in an attempt to gain a level to re-enter the Wizard Competition . Special guest star : Gregg Sulkin as Mason Guest stars : Andy Kindler as Chancellor Tootietootie , Frank Pacheco as Felix , David Barrera as Carlos Cucuy , Roxana Brusso as Julie Cucuy , Samantha Boscarino as Lisa Cucuy , Kari Wahlgren as Helen 82 `` Lucky Charmed '' Victor Gonzalez Justin Varava December 10 , 2010 ( 2010 - 12 - 10 ) 403 3.0 Harper tries teaching Alex how to live without magic , but the strain of taking responsibility for her actions instead of using magic causes a disagreement between the two best friends when Alex gets in big trouble for crashing Mr. Laritate 's car . As punishment , Alex has to clean the school 's garbage cans . Mr. Laritate forgives Alex after seeing how the crash influenced her , and Alex realizes that she will be okay without magic . Wanting to thank Harper , Alex uses Mr. Laritate 's lucky life - saving hula girl figurine , Leilani , for luck , as Mr. Laritate does . During Harper 's driving test , Alex turns into the hula girl and guides Harper through the test , knowing that she might fail . In the end , Harper passes her test . Meanwhile , Jerry wants to pass on the wizard family robe to Max , but Justin has a hard time letting go of it . Guest stars : Bill Chott as Mr. Laritate , Julio Oscar Mechoso as DMV Instructor Absent : Maria Canals Barrera as Theresa Russo 83 `` Journey to the Center of Mason '' Robbie Countryman Peter Dirksen December 17 , 2010 ( 2010 - 12 - 17 ) 404 N / A After she finds out that she will lose her powers , Alex can no longer date Mason , so they decide to try being best friends instead . After watching a film at the theaters , Harper , Alex , and Mason go home , Dean , who was Alex 's Ex-Boyfriend comes and asks Alex to get back together with him . Mason gets jealous . When Dean brings a gift for Alex , Mason then asks what Alex 's relationship is with Dean , and Max tells him that Dean and Alex used to date , making Mason more jealous . Mason then turns into a werewolf and eats Dean . Alex , Justin , and Max come up with a solution to turn Justin 's toy Captain Jim Bob submarine big and go in it . They then shrink it and go into Mason 's mouth . While there , Alex , Max , and Justin see Mason 's thoughts , which are all of Alex . They get Dean out , and Alex , when Harper , Max , and Justin leave , Alex tells Dean that he was knocked out and the doctor 's solution was for them all to dress up as cows . Dean then gives her his offer once again : he wants to get back with her . Alex tells him ' no ' , and then gets back together with Mason , with deciding to get back to the Wizard Competition . Special guest star : Gregg Sulkin as Mason Guest stars : Daniel Samonas as Dean , Frank Pacheco as Felix , Shoshana Bush as Tyler 84 5 `` Three Maxes and a Little Lady '' Victor Gonzalez Richard Goodman January 7 , 2011 ( 2011 - 01 - 07 ) 405 4.0 Alex joins Justin 's wizard class to try to move up in the competition . Max joins a sophisticated wizard society . They suggest he move the wizard competition up and he moves it to a few days away . Max quits the society because he finds out that they are taking advantage of him . Alex and Justin transform into Max in an attempt to change the date of the family wizard competition back to its original date and succeed . In the end of the episode they accidentally turn Max indefinitely into a little girl . Guest stars : Bailee Madison as Maxine , Frank Pacheco as Felix , Kari Wahlgren as Helen , Shane Harper as Fidel Note : Bailee Madison begins her recurring role as Maxine , the female Max in this episode . Jake T. Austin was absent in four other episodes and returns at the end of the tenth episode `` Back to Max '' . 85 6 `` Daddy 's Little Girl '' Robbie Countryman Gigi McCreery & Perry Rein January 21 , 2011 ( 2011 - 01 - 21 ) 406 3.4 With Max still a little girl , their parents pay more attention to Max ( Maxine ) , and Alex becomes jealous . Justin takes Maxine to her ( Max 's ) karate class and Maxine kicks Justin 's butt . After the class , Alex decides to make Jerry 's favorite sandwich in order to win his love back , but Maxine comes along with Justin , announcing that she beat Justin in karate , and Jerry immediately falls for her cuteness again . More jealous than ever , Alex comes back with Justin to Maxine 's karate class with their parents , having a slight idea of knowing how to change Maxine back to Max . They cast spells on Maxine , Justin using the spell he used on Max that turned him into Maxine , and Alex using the reverse spell . Unfortunately , it only makes Maxine cuter , which makes her furious . Instead of going against the person she was supposed to , she demands that she kicks either Alex or Justin 's butt . Justin goes against her , but seeing that he would lose , Alex casts a spell on Justin to make him a karate master . Too fast for Maxine , Justin beats her , and Maxine is hurt ( physically ) . Jerry and Theresa rush to her side , caring for Maxine more than they care that Justin beat Maxine in karate . Justin leaves , and Alex leaves after him , heartbroken . Back at home , Jerry tells Alex that he fell for Maxine because she reminded him of Alex when she was little and they bond again . Meanwhile , Zeke thinks the Russos hate him , but really it was just Harper 's alibi in order to keep him away from the Sub Station so as not to make him suspicious about Max 's disappearance and Maxine 's appearance . Guest stars : Bailee Madison as Maxine , Dan Benson as Zeke , Lanny Joon as Sensei Absent : Jake T. Austin as Max Russo 86 7 `` Everything 's Rosie for Justin '' Victor Gonzalez Justin Varava February 4 , 2011 ( 2011 - 02 - 04 ) 407 3.9 Justin and his delinquent class have a wizard wand dance assessment to do which will get them back in Wiztech . A new student named Rosie joins , whom Justin falls madly in love with to later find out that she is an Angel . Meanwhile , Jerry , Teresa , and Harper try to think of ideas of how to get more people to come to their restaurant . Harper makes cards which customers get hole punched each time they buy a sandwich ; when you get 9 hole punches , they get a free sandwich . Contrary to the original purpose , one man punches his own card , demands a free sandwich , and then brings dozens of people into the restaurant , who also have pre-punched cards . Maxine uses her charming cuteness to convince the customers to not exploit her parents . Meanwhile , Justin kicks Rosie out of his class so that it does not fail the wizard dance assessment . Sympathizing , Alex gets Rosie to rejoin the class and puts a copycat spell on her so that Rosie can mimic Alex 's move and pass the class . The assessment ends in disaster when a squirrel - frog attaches itself to Alex 's foot , and Rosie reveals herself to be an angel . The class fails , but Justin starts dating Rosie . Guest stars : Bailee Madison as Maxine , Frank Pacheco as Felix , Leven Rambin as Rosie , Diane Delano as Penny , Chris Coppola as Customer , Cameron Sanders as Nelvis Absent : Jake T. Austin as Max Russo 87 8 `` Dancing with Angels '' Victor Gonzalez Richard Goodman February 11 , 2011 ( 2011 - 02 - 11 ) 408 4.5 Justin and Rosie start dating but when their date gets ruined , they want a second date . Alex suggests an Angel Club in Los Angeles , but Rosie says that only Angels are allowed . Maxine will not tell their parents as long as Alex brings her back an Ozzy Osbourne stone . After Justin , Harper and Alex sneak in along with Rosie , Zeedrik asks them if they are angels and they say they are . Back at the sub shop , Theresa and Jerry question Maxine to find out where Justin , Alex and Harper are , and Maxine does not say , so they force her to join a Pageant in order for her to tell them where they are . Things get bad at the angel club when Alex and Harper can not fly , until Justin secretly pulls out his wand and gives Harper flying powers . Zeedrik still questions them by giving them a test to sing while playing the harp . Though Alex , Justin , and Harper fail to do so , they are accepted as `` angels '' until the key - chain of Ozzy Osbourne 's star reverts to normal . Zeedrik believes they are really angels of darkness , causing everyone in the club to run off , even Rosie . Rosie comes back at the end of the episode to comfort Justin , but when they both look over the city , Rosie 's wings are shown to be black , the color of an angel of darkness ' wings , but Justin does not see because he is looking forward . Meanwhile , Maxine ends up winning the beauty pageant , making Jerry and Theresa celebrate . Guest stars : Bailee Madison as Maxine , Leven Rambin as Rosie , Ransford Doherty as Zeedrik Absent : Jake T. Austin as Max Russo 88 9 `` Wizards vs. Angels '' Victor Gonzalez Vince Cheung & Ben Montanio February 18 , 2011 ( 2011 - 02 - 18 ) 409 -- 410 5.1 Rosie is an Angel of Darkness , and has been tricking Justin all along to make him evil . Alex finds out , with the help of Tina ( a guardian angel in training trying to earn her wings ) that Rosie is an Angel of Darkness . Justin crosses over to the dark side with Rosie , steals the `` moral compass '' , and gives it to Gorog , so Gorog can turn the compass from good to bad , so that the world will be covered in darkness and every human being will be corrupted . Rosie tries to get Justin to run away and save him , because Gorog wants to destroy him , but Justin refuses to leave , stating that he is no longer a wizard but an Angel of Darkness and breaks his wand in halves . Alex tries to convince Justin that he is a wizard , a good wizard and not an Angel of Darkness . Meanwhile , Maxine is enrolled in Tribeca Prep , since she is participating in a `` cousin exchange program . '' After her first day of school , Maxine laments that she has no friends , which leads Harper to insist Maxine have a slumber party ( which is really what Harper wants because she has never had a slumber party ) , that eventually becomes a disaster . Tina reveals that Rosie was a good angel , and her teacher however , Gorog reached her before her transport arrived and corrupted her . Alex and Tina go to the Dark Realm to save Justin . Alex borrows a pair of wings from the guardian angels . Alex takes the Moral Compass and tries to turn it to good but Justin stops her and then Rosie used her good angel side and they fly up and when they land , Justin turns the compass to good . Tina and Alex take it and run . When Rosie and Justin kiss , Rosie turns into a good angel ( with white wings ) , and Justin loses the dark wings . Guest stars : Bill Chott as Mr. Laritate , Bailee Madison as Maxine , Leven Rambin as Rosie , John Rubinstein as Gorog , China McClain as Tina , Amy Tolsky as Florence Note : This is a one - hour special . Absent : Jake T. Austin as Max Russo 89 10 `` Back To Max '' Guy Distad Todd J. Greenwald March 11 , 2011 ( 2011 - 03 - 11 ) 411 3.7 Justin invites Professor Crumbs to check on his delinquent wizards class , to see how their wizard training is progressing . During the conversation , Maxine interrupts , causing Alex and Justin to panic since they do not know how to explain `` Maxine '' , and they try to find a way to turn Maxine back to Max . Meanwhile , Harper takes a role in the school 's `` Spirit of America '' play , only to be replaced by Maxine . Guest stars : Bill Chott as Mr. Laritate , Ian Abercrombie as Professor Crumbs , Bailee Madison as Maxine , Cameron Sanders as Nelvis , LJ Benet as Little Crumbs , Jackie Evancho as Choir Girl Note : Jake T. Austin returns as Max near the end of the episode . Bailee Madison marks her final appearance as Maxine in this episode . 90 11 `` Zeke Finds Out '' David DeLuise Peter Dirksen April 8 , 2011 ( 2011 - 04 - 08 ) 412 3.0 When Alex gets bored of Zeke 's failing magic tricks , she decides to use magic to make Zeke 's tricks come true . Everything goes downhill when Zeke tells Harper he believes he is a wizard and is now booking a magic act at the sub station for a kid 's birthday party . Since Justin and Harper are tired of Alex ruining things all the time , they decide to leave everything up to her . Harper , not wanting to let Zeke make a fool of himself , tells Alex to tell Zeke the truth about wizards , but she is reluctant to , as exposing wizardry to another mortal could get her kicked out of the wizard competition forever if anyone finds out . Zeke goes on with his belief that he is a `` wizard '' , but he finds out Harper is keeping a secret from him so he breaks up with her . Alex realizes what a mess she created and on the day of Zeke 's magic act , she makes Harper disappear before she can expose wizardry to Zeke , which freaks him out . The Russos really are in deep waters when Zeke is going to saw Max in half with a real saw for his next act , so Justin takes matters into his own hands and freezes everyone except for himself and Zeke . Justin tells Zeke that he is a wizard , not Zeke , and that he did everything . Still not believing him , he flashes Zeke into their lair and unveils the secret to him . They go back outside and tell Harper and Alex who are fighting , that Zeke now knows about the existence of wizards , but he promises he will never tell anyone . Zeke and Harper reconcile their relationship and Alex and Harper apologize and make up with each other . Guest stars : Dan Benson as Zeke , Maurice G. Smith as Big Mitch Absent : Maria Canals Barrera as Theresa Russo , David DeLuise as Jerry Russo 91 12 `` Magic Unmasked '' David DeLuise Justin Varava May 13 , 2011 ( 2011 - 05 - 13 ) 413 2.7 When a former legendary Luchador `` Muy Macho '' visits the Sub Station , Alex finds out that she was the reason why he quit lucha libre ( wrestling ) , so she tries to make it up to him by arranging a lucha libre match with him and Jerry to restore his honor . Meanwhile , Zeke asks too many questions about wizardry , so Justin offers to grant him one magical wish if he promised he would stop questioning him about magic . Unfortunately , Zeke 's wish is to defeat Muy Macho in the match . Max pretends to like Talia 's hobbies , but does not enjoy them . Guest stars : Dan Benson as Zeke , Valente Rodriguez as Muy Macho , McKaley Miller as Talia 92 13 `` Meet the Werewolves '' Victor Gonzalez David Henrie June 17 , 2011 ( 2011 - 06 - 17 ) 415 4.0 When Alex asks Mason if she can meet his parents , Mason shows his fake parents and she finds out soon after . She gets upset at Mason and demands him to take her to his real parents . Mason takes her to the Autumn Moon Feast ( werewolf holiday ) to meet his parents . Soon after that Alex finds out he told his parents that she is a werewolf and is frustrated . She uses a spell to change herself and Harper into werewolves . She tries to be the worst werewolf possible in front of Mason 's parents during the feast . In the end Mason 's parents like her as a werewolf and she says to him that they do not know the real her . She begins to leave and he says that he truly loves Alex to his parents and that she is a wizard . She changes back to a wizard and Mason turns back to human form . After the parents find out they try to eat her and Mason and Alex make a near escape . Meanwhile Max makes his parents ' food taste like kid food in order to make it taste good . Afterwards Jerry and Theresa eat it , and as a result they act like little kids . Justin thinks he will have to fix it but Max comments on how he does not need help , as he will be the future family wizard . Max uses another spell on cookies to make them older , but they turn into teenagers . Finally , Justin uses a spell on a pizza to turn them back into their regular selves . Special guest star : Gregg Sulkin as Mason Guest stars : Harry Van Gorkum as Grant Greybeck , Robin Riker as Linda Greybeck , Marianne Muellerleile as Molly , Eric Zuckerman as Bill 93 14 `` Beast Tamer '' Victor Gonzalez Gigi McCreery & Perry Rein June 24 , 2011 ( 2011 - 06 - 24 ) 416 4.0 When Alex , Justin and Max win four tickets to the `` Beast Bowl '' , the tickets are delivered by Beast Tamer Chase Riprock , who begins flirting with Alex and invites the group for a pre-bowl tour . Alex invites Mason to go with them but he says he can not . During the tour , despite Chase 's warnings , Justin attempts to tame the beast so that Max can have his photo taken with the beast , but both are caught and banned from the event . Chase tries to kiss Alex but she rejects him and returns to the mortal world where she finds Mason coming out a hardware store . Mason is evasive about his plans , which involve making an anniversary present for Alex , so Alex returns to the Beast Bowl. Harper tells Mason that Chase likes Alex and urges him to surprise her with the gift at the Beast Bowl . Meanwhile , Justin and Max return to the Beast Bowl disguised as beast taming clowns where they allow the beast to escape and snatch Alex who throws the whip to Chase . Chase tames the beast who drops Alex and Mason is able to reveal his gift , which is a sculpture of himself and Alex. Alex and Mason leave together , with the sculpture . Special guest star : Gregg Sulkin as Mason Guest star : Nick Roux as Chase Absent : Maria Canals Barrera as Theresa Russo , David DeLuise as Jerry Russo 94 15 `` Wizard of the Year '' Victor Gonzalez Richard Goodman July 8 , 2011 ( 2011 - 07 - 08 ) 417 3.6 Alex has been crowned Wizard of the Year after saving the world from dark angels , as well as being reinstated in the Wizard Competition , and a banquet honoring her achievement is hosted . With paparazzi in tow , Chase Riprock visits the lair to congratulate Alex . As he leaves , Keith Keith hounds them about being in a romantic relationship , to which Alex states she already has a boyfriend . Later , Mason and Alex watch his gossip segment , and to their horror , Keith Keith confirms that Chase and Alex are dating . For Alex winning Wizard of the Year , the Russo family must record a holographic video congratulating Alex . However , much to Justin 's displeasure , Justin 's video instead only contains him complaining about Alex . The night of the awards banquet , Mason stands Alex up , leaving her devastated . At the banquet , Chancellor TootieTootie tells Justin that his display of affection and pride for Alex may reinstate him into the competition , causing Justin to be frantic and do everything within his power to destroy the tape . Chase purposely invites himself to sit at the Russo 's table . However , Mason shows up late and sees Alex and Chase flirting together . A short battle ensues between Mason and Chase . Alex soon breaks up the fight , and is forced to choose either Mason or Chase . Appalled by both of their behaviors that evening , Alex chooses neither and breaks up with Mason . Special guest star : Gregg Sulkin as Mason Guest stars : Andy Kindler as Chancellor Tootietootie , Nick Roux as Chase , Justin Guarini as Keith Keith 95 16 `` Misfortune at the Beach '' Jody Margolin Hahn Vince Cheung & Ben Montanio July 24 , 2011 ( 2011 - 07 - 24 ) 420 3.5 The Russos head to the beach on an extremely hot day , and Harper hopes to enjoy reading her book while at the beach . The Russos come across Zelzar , a fortune - telling machine . Jerry warns the kids about Zelzar -- he is from the wizard world , and a wizard 's fortune really comes true . After Max and Justin receive good fortunes , Alex decides to get a fortune despite her father 's warning . However , her fortune says , `` Say good - bye to your life . '' Alex thinks nothing of it and proceeds to enjoy the day at the beach , but comes close to life - threatening situations multiple times . Alex , Justin , Max , and Harper make a compromise with Zelzar -- if they can extract him from the machine and let him enjoy a day at the beach , he will take back Alex 's fortune . Max replaces Zelzar , and Justin is forced to drive the beach - goers away from the machine ( as Max gives out silly fortunes about his new facial hair ) ; meanwhile , the girls help Zelzar fancy his day at the beach . In the end , Zelzar reveals that while he can not take away Alex 's fortune , he can give it to the next person . The next person happens to be a little girl , who gets Alex 's fortune . However , it is revealed that a man from Random Prize Giveaway shows up , saying : `` Say good - bye to your life because we 're randomly giving you one million dollars ! '' Guest stars : Michael Carbonaro as Zelzar , John J. York as Game Show Host 96 17 `` Wizards vs. Asteroid '' Victor Gonzalez Peter Dirksen August 19 , 2011 ( 2011 - 08 - 19 ) 418 3.6 When it is announced that a huge asteroid is hurtling towards the earth , the Russos decide to hide in the wizard world , but Alex , Justin , and Max change their minds and go into space to destroy the asteroid . Meanwhile , Alex finds out that she is not graduating high school , but when a small part of the asteroid hits the school , she gets a chance to graduate . Guest stars : Dan Benson as Zeke , Bill Chott as Mr. Laritate , Kristina Hayes as Reporter 97 18 `` Justin 's Back In '' Guy Distad Todd J. Greenwald August 26 , 2011 ( 2011 - 08 - 26 ) 419 3.2 Justin 's class of delinquent wizards has completed their studies , which means they are eligible to rejoin WizTech if they all pass their final evaluation . When they all fail , Alex suspects something is not right . The truth soon uncovered -- the class had actually passed , but the historian who checked the exams was revealed to be evil , and had lied about the exam scores . He tries to attack Justin 's class , so Justin must stop the historian , and save Alex. but one of the students in the class is able to stop him by pulling the most powerful wand in the wizard world out of a crystal ball and using it . This means that he is actually the descendant of a famous wizard who had previously been thought to have no living descendants . Justin 's class is revealed to have passed the final exam , so they are readmitted into WizTech and Justin is officially back in the family wizard competition . Meanwhile , Jerry and Theresa realize that they never saved any of Max 's childhood accomplishments , so they try to recreate them with the help of Harper . During the credits , it is revealed that Max had kept his favorite memories all along . Special guest star : Tim Conway as Cragmont Guest stars : Ian Abercrombie as Professor Crumbs , Frank Pacheco as Felix , Cameron Sanders as Nelvis 98 19 `` Alex the Puppetmaster '' Victor Gonzalez Gigi McCreery & Perry Rein September 16 , 2011 ( 2011 - 09 - 16 ) 421 3.0 Alex and Harper decide to do a puppet show to raise money for a new apartment and Zeke and Justin help them . But Alex forgets to write the lines for the show . Harper gets mad and decides to do a show of her own and makes her puppets act like the way Alex treats her . Alex makes Justin and Zeke puppets for her show and steals Harper 's crowd which makes Harper upset . Then Alex apologizes to Harper using the puppets . Meanwhile , Talia 's parents meet Theresa and Jerry for the first time , but it does not go well . Justin puts a spell on Talia 's parents to forget they ever meet Theresa and Jerry . They meet again , but this time they end up watching a 16 - hour play together . Special guest star : Joely Fisher as Meg Robinson Guest stars : Dan Benson as Zeke , James Urbaniak as Rob Robinson , McKaley Miller as Talia 99 20 `` My Two Harpers '' Victor Gonzalez Manny Basanese September 30 , 2011 ( 2011 - 09 - 30 ) 422 3.0 Harper and Zeke spend so much time together that Alex decides to make a Harper - clone . Justin comes with lots of different ideas to save money in the sub-shop and Max starts a fast - food restaurant in the wizard world . Guest star : Dan Benson as Zeke Absent : Maria Canals Barrera as Theresa Russo 100 21 `` Wizards of Apartment 13B '' Guy Distad Justin Varava October 7 , 2011 ( 2011 - 10 - 07 ) 423 3.5 Alex and Harper receive a flyer for a hotel . They meet the hotel 's owner , Dexter , who sells them Apartment 13B , which belongs to a floor for non-humans . They rent it , but Harper is forced to use a training wand that can only open things . Later at their apartment , they realize Mason lives on the floor . Mason will stop at nothing to win Alex back ; Alex tells him she does not want a boyfriend , but Mason is relentless . That night , Alex and Harper throw a party , where the whole floor comes , including Felix . Mason goes to the party and uses Harper 's wand to open Alex 's heart . However , she flirts with an ugly giant . Mason tries again , but makes Alex opens her heart to Felix . Meanwhile , Theresa and Jerry trick Justin into going to Alex 's apartment to drive him out of the house . When Justin arrives , he fixes Mason 's mistakes by resetting Harper 's wand and it is revealed that despite his actions , he misses Alex. Alex , freed from Mason 's spell , is angry at Mason , and Mason , deflated , leaves . Outside their door , Dexter turns into Gorog when no one is looking . He wants revenge . Max appears in the apartment , and when Harper and Alex kick him out , he encounters Gorog , who gives him a flyer for a free camp . Max accepts . Special guest star : Gregg Sulkin as Mason Guest stars : John Rubinstein as Gorog , Frank Pacheco as Felix , Fred Stoller as Dexter , Christopher Douglas Reed as Ogre Note : This is part 1 of a 4 - part arc and also the 100th episode . 101 22 `` Ghost Roommate '' David DeLuise Richard Goodman October 14 , 2011 ( 2011 - 10 - 14 ) 424 4.1 Alex and Harper have trouble paying the apartment bills , so Dexter recommends getting a roommate . Alex and Harper find a wealthy ghost named Lucy , who they let move in . Meanwhile , Justin orders a robot to help Zeke at the sub station , as Justin is busy studying for the wizard competition . The robot is lost in transit , but Dexter ( Gorog in disguise ) makes an evil robot to spy on Justin to pass information to Gorog . While Zeke and the robot are working , the robot asks him about Justin and the wizard competition , which confuses Zeke because no one had talked about it in front of the robot . Later that night , Lucy starts crying and explains that her boyfriend Donny had disappeared sixty years prior in a plane accident . Alex tries to set Lucy up with Justin , but he makes matters worse . Later , Lucy meets Mason and they go out on a long date , which makes Alex jealous . Meanwhile , the robot again asks Zeke about the wizard world and Zeke realizes that the robot might have a secret agenda . He tells Justin , who simply does not believe him and is ignorant of the situation because of his intensive studying . Because of her jealousy , Alex helps Lucy locate Donny , who had been stranded on an island in the Bermuda Triangle . They soon reconnect , but not after Alex realizes that she can not get back home , due to heavy magnetic currents interfering with her magic . Mason goes to save Alex after he finds out that she had left for the Bermuda Triangle . The robot attacks Justin and Zeke after they find him scanning the spell book . Mason and Alex manage to get off of the island using the Bermuda shorts given to them by Justin . The robot is shown giving Gorog the spell he needed that he got from the Russos . Special guest star : Gregg Sulkin as Mason Guest stars : Dan Benson as Zeke , Fred Stoller as Dexter , Josh Sussman as Hugh Normous , Linsey Godfrey as Lucy , Travis Caldwell as Donny , Michael Carbonaro as Robot Note : This is part 2 of a 4 - part arc . Absent : Jake T. Austin as Max Russo , Maria Canals Barrera as Theresa Russo , David DeLuise as Jerry Russo 102 23 `` Get Along , Little Zombie '' Victor Gonzalez Peter Dirksen October 21 , 2011 ( 2011 - 10 - 21 ) 425 3.4 Alex and Harper are going to their apartment when they run into Mr. Laritate , who also lives in the hotel . Felix tells Justin he broke his wand and Justin informs him that it 's fake , because he saw batteries in it . He concludes that someone stole Felix 's wand . Dexter is showing a zombie to the hall for magical creatures while he introduces them to Alex and Harper . Mason , who will stubbornly not stop asking Alex on a date , meets them in the elevator . When Alex ignores him , Mason messes with the buttons so he can spend more time with her and they land on the second floor . Justin tells Jerry that Felix lost his wand and Jerry tells them to use the abracadoobler wand app and Felix realizes that his wand is around the area . Meanwhile , because Mason messed with the buttons , all the humans have discovered the mysterious thirteenth floor . Alex lies and tells the humans that they accidentally got into a secret open house floor for the people in the hotel . Mr. Laritate , after all the other humans left , discovers the thirteenth floor too and sees Alex , Harper , and Mason . Mr. Laritate sees the zombie , Alex explains this is a haunted house , but Mr. Laritate does not believe them . The zombie bites Mr. Laritate and turns him into a zombie . They take him to their apartment while Alex looks for a spell to turn Mr. Laritate human again . Mr. Laritate runs away and Alex , Harper , and Mason suspect he went to the substation . Professor Crumbs questions Felix , who reveals the 13th floor to them , and Crumbs denies the existence of a `` 13th floor '' , so they go to investigate . At Waverly Place they 're throwing a festival while Alex tells Jerry what happened . Alex and Mason , trying to sneak the zombie into the substation , are forced into dance . They then , after Harper causes a distraction , take Laritate to the substation and give him a medicine from the wizard world that turns Laritate human . He remembers nothing and goes outside to dance . Alex kisses him on the cheek , and Mason suggests they just be friends . Alex gets back together with Mason . Crumbs tells Felix the floor should not exist . Harper , Mason , and Alex return to the thirteenth floor . Justin tells them that the floor does not exist . Crumbs thinks that there is evil going on . Dexter then comes with dark angel wings and it is revealed he stole Felix 's his wand . Dexter tricks Felix into turning to the dark side in return for the wand . Felix casts a spell that traps them all on the floor . Back at Waverly Place , Mr. Laritate is dancing while Jerry is selling chili , when Mr. Laritate gets one and gets water to get `` the burn '' out of his mouth . Special guest star : Gregg Sulkin as Mason Guest stars : Bill Chott as Mr. Laritate , Ian Abercrombie as Professor Crumbs , Frank Pacheco as Felix , Fred Stoller as Dexter , Regan Burns as Everett , Jane Carr as Martha St. Claire Note : This is part 3 of a 4 - part arc . Absent : Jake T. Austin as Max Russo , Maria Canals Barrera as Theresa Russo 103 24 `` Wizards vs. Everything '' Victor Gonzalez Gigi McCreery & Perry Rein October 28 , 2011 ( 2011 - 10 - 28 ) 426 3.4 The wizards of 13th floor come out and Alex explains Dexter 's evil . Dexter reveals himself as Gorog , and that he is trying to rebuild his army of darkness . Felix uses a spell that makes all the wizards from the 13th floor go with them , so they can make them all evil , all except Alex , Justin , and Harper , who are left behind . A trashcan tells them to go through it , however Alex ends up in the room again . Justin realizes only mortals can go through it . Alex convinces Harper into going through it , Harper then arrives in the wizard lair . Gorog takes the kidnapped wizards to his evil lair , where he plans to make them evil . He tricks Hugh Normous into becoming one of the evil wizards in his army . Professor Crumbs fakes becoming evil when Gorog plans on making him evil . Harper contacts Max and tells him about Gorog . Max arrives and suggests they look for spells in the wizard books . Gorog makes the wizard army to dig a hole to the atmosphere to the wizard world , so they can take over it . Gorog also tries to trick Mason into being one of the members in his army , but Mason refuses . Gorog tricks him into joining , however . Gorog then makes Felix send a black whole to suck Alex and Justin in it . Gorog gives the kidnapped wizards their wands so they can take over the wizard world , but Crumbs refuses to let them . Felix then gets rid of Crumbs on Gorog 's order . The black hole nearly sucks Alex and Justin when Crumbs arrives at the wizard lair . Max suggests they put a black hole in the wizard lair . Max goes in and rescues Alex and Justin . Crumbs tells them to use their magic to stop Gorog . They arrive at Gorog 's lair and attempt to destroy him , but Mason tries to convince Alex to join them . Gorog also reveals Juliet ( young again ) , Justin 's ex-girlfriend , and she convinces Justin to join . Alex also surrenders , and Max is forced to as well , the three then use their combined magic to get rid of Gorog , and all the supernatural beings are freed from his influence . Juliet and Justin get back together . Alex tricks Harper into moving in the Russos ' basement by making her think all the wizards and supernatural creatures , with no home , are going to stay . Special guest stars : Gregg Sulkin as Mason , Bridgit Mendler as Juliet van Heusen Guest stars : Ian Abercrombie as Professor Crumbs , John Rubinstein as Gorog , Frank Pacheco as Felix , Fred Stoller as Dexter , Josh Sussman as Hugh Normous , Christopher Douglas Reed as Ogre Note : This is part 4 of a 4 - part arc . Absent : Maria Canals Barrera as Theresa Russo , David DeLuise as Jerry Russo 104 25 `` Rock Around the Clock '' Guy Distad Robert Boesel November 4 , 2011 ( 2011 - 11 - 04 ) 414 3.8 The current landlord of the Russo 's building decides to sell the sub shop , evicting the Russos of the shop , as well as their apartment and lair , which contains their portal . Without the portal , the Russos will lose all contact with the Wizard world . So , the entire family ( along with Harper ) travel back to 1957 to stop Jerry 's father ( the previous owner of the shop ) from selling the shop to the landlord , so that they do not get evicted . Jerry 's father agrees not to sell the shop , and the Russos return to present day . However , they accidentally leave Harper behind when Harper is distracted by a woman with a poodle , and Harper suggests that the woman should put a picture of a poodle on her skirt , creating a poodle skirt , which were popular in the 1950s . Back in present day , the Russos ( who have not yet realized Harper 's absence ) find the sub shop obsolete , boarded up , and empty , with the subway car and the lair missing , too . The Russos realized that they have messed with the fabric of time , and something they did while time - traveling in 1957 must have created a ripple effect , and affected present day . While Alex suggests that they go back to 1957 to find out what they did wrong , Justin denies this , saying they should not solve the problem by doing exactly what caused the problem , however Theresa states that they will have to go back , when she realizes that they left Harper in 1957 . The Russos return to 1957 , where Jerry 's father reveals that he did not sell the restaurant to the landlord , however the restaurant failed to make enough business , and that he has been forced to close it . Justin realizes that when Jerry 's father lost the restaurant , he moved out of the building and since their family no longer lived in the building , the lair disappeared in present day . The Russos decide just to find Harper and go back to present day , and discover that Harper enrolled at Tribeca Prep as soon as they forgot her in 1957 , as Harper did not want to ruin her perfect attendance record . Justin , Alex , and Max go to Tribeca Prep to retrieve Harper , while Jerry and Theresa wait at the sub shop , however Harper decides to stay in the past , as she finds new popularity in 1957 that she does not experience in present day . While at Tribeca , Max introduces the high - five to several students . Alex realizes that if they make the sub shop the 1957 high school hangout , the sub shop will get better business and will not have to close . However , no one has any fun at the sub shop that night as the jukebox is broken . Just as everyone is about to leave , Max plugs his MP3 player into the jukebox and Harper uses her popularity to get the students dancing and having fun at the shop , and the business at the sub shop peaks . The Russos return to the present day , where the lair and the sub shop are back , and Jerry owns the entire building , while the landlord is now the owner of a Janitor service , instead of the building , and the high - five is now called a `` Max '' . Alex then remembers that they forgot Harper back in 1957 and they go back and get her . ( the Russos accidentally end up in 1977 twice in the episode , thanks to Justin ) Guest stars : Brendan Bradley as Hank , Carlo Michael Mancini as Lenny Hune 105 26 `` Harperella '' Victor Gonzalez Richard Goodman & Justin Varava November 18 , 2011 ( 2011 - 11 - 18 ) 427 3.5 The episode begins on Waverly Place where Harper is reading a storybook to a group of children when Alex pushes her away . Alex explains that it 's a Wizard Storybook , a magical storybook that has the power of turning what the person reads real . Then Harper is teleported into the loft where Theresa , her evil ( or `` wicked '' as she referred ) stepmother calls her Harperella ( a combination of Harper and Cinderella ) and forces her to do the chores . Then her evil stepbrothers , Justin and Max come down and give her laundry to her wash . Later Harper is washing the Waverly Substation when Alex appears , as she is Harper 's fairy godmother . Alex explains to Harper that the fairy tale has to run as it 's supposed to before things can go back to the way they were and then vanishes . Then Max comes in and says that the prince charming has decided to use the Substation to be the ballroom for the royal party . Harper says that she can go to the party but her stepfamily laughs at her . Back in the loft Harper summons Alex , she uses a spell to transform Harper 's bookdress into a princess gown and glass bowls into glass slippers . They go down ( through the door ) into the street and Alex turns the hot dog stand into a carriage to take Harper to the party . There she dances with prince charming , Zeke . At the stroke of midnight she goes outside leaving her glass slipper behind . Then , to both girls ' shock , Justin , Max and Zeke ( with pig noses ) rush inside the substation being chased by the big , bad wolf , Mason . Alex realizes that some pages are missing and Cinderella and three little pigs stories were blended together . The pig - turned boys and Harper rush into the loft being chased by Mason . While Alex finds the pages Harper has to keep Mason from eating Zeke ( who built the straw house ) , Alex eventually finds the book pages in a drawer along with other book pages . Alex tries a page from Max 's favourite magic trick book then David Copperfield comes into the room and does some magic tricks which impresses Harper . But another page that Alex tried was from Justin 's science book and a caveman appeared , but the caveman has a thing for Harper . So when Harper is impressed by the magic trick he is jealous and tries to attack David Copperfield . Alex finds the right pages and the story goes back to Cinderella . Zeke Charming comes in and Theresa tries to wear the slipper but fails . Then Harper tries it and fits . She and Zeke go back to the royal party where they dance and then the story ends and the world is returned to its normality . Last but not least , Harper was back in the real world and she lived happily ever after . Finally , Alex planned that Harper and Zeke dance because they were great when they danced in romance , so they practiced but they were really bad . The ball dance is turned into a clogging dance . Special guest stars : David Copperfield as himself , Gregg Sulkin as Mason Guest star : Dan Benson as Zeke Absent : David DeLuise as Jerry Russo 106 27 `` Who Will Be the Family Wizard '' Victor Gonzalez Vince Cheung & Ben Montanio ( Part 1 ) Todd J. Greenwald ( Part 2 ) January 6 , 2012 ( 2012 - 01 - 06 ) 428 -- 429 9.8 During a dinner that Alex made for the family , Professor Crumbs visits to reveal that with this selfless act , the Russo children can finally have their family wizard competition . Round one of the competition is a trivia round , in which questions related to magic are asked . During this round , Harper and Zeke appear . Zeke has accidentally smelled a purple substance in the lair that caused him to turn purple . This causes a griffin to kidnap him and Harper . Alex , Justin and Max use their three time - outs to go and rescue their friends . They make it back , only to discover that they have been gone too long and have been disqualified from the competition . They all return home , where the lair disappears and they all lose their magic , thus becoming mortals . Justin and Max begin to hate Alex because of her insistence on saving Zeke and Harper resulting in them losing . Convinced that the loss of their powers has ruined the family , Jerry decides to sell the sub shop . The kids go for a day without magic , during which , slowly gaining back trust in each other as they reopen and run the shop . Suddenly , they are returned to the area of the Wizard Competition , where the host reveals that the griffin attack ( which was actually meant for Professor Crumbs ) and the weeks they went without magic were tests to test their family bond . They enter the final round of the competition , a massive labyrinth . The first out of the maze will win . Justin makes it out first and is declared the Family Wizard . However , he decides he can not accept this because Alex was about to get out first : Justin had been trapped by magical vines and Alex , near the exit , came back to help . Alex is thus declared the true family wizard . Professor Crumbs , proud of Justin 's integrity in telling the truth , declares that he is going to retire as Headmaster of Wiz - Tech and declares Justin the new headmaster , granting him full wizard powers . Considering Max is the only Russo child that does not get powers or becomes a full wizard , Jerry thus decides to one day pass down the sub shop to Max , which he happily accepts . It ends with the Russo family hugging and gathering together with Alex stating that they are all finally happy at the same time . Special guest stars : Gregg Sulkin as Mason , Bridgit Mendler as Juliet van Heusen Guest stars : Dan Benson as Zeke , Ian Abercrombie as Professor Crumbs , Andy Kindler as Chancellor Tootietootie Note : This is an hour long special .
  Fact 0.0: The Wizards of Waverly Place season 4 episode list gives each episode's number, title, directors, writers, original air date, production code and U.S. viewers in millions.
  Fact 0.1: Episode 80, 'Alex Tells the World', was directed by Victor Gonzalez and written by Gigi McCreery and Perry Rein.
  Fact 0.2: Episode 80 aired November 12, 2010, with production code 401 and 4.1 million U.S. viewers.
  Fact 0.3: In 'Alex Tells the World', the Russos and Mason make it back home but their wizard powers are still gone.
  Fact 0.4: In 'Alex Tells the World', Alex reveals to reporters that they are wizards and that a group of wizards is held captive by the government.
  Fact 0.5: In 'Alex Tells the World', Professor Crumbs tells Alex that everything was a test she failed and orders her to Wizard Court for exposing wizardry.
  Fact 0.6: In 'Alex Tells the World', Justin is ordered to Wizard Court because he exposed magic to Agent Lamwood.
  Fact 0.7: Agent Lamwood was a creation from Professor Crumbs' mind.
  Fact 0.8: In 'Alex Tells the World', the judges rule Alex and Justin guilty.
  Fact 0.9: In 'Alex Tells the World', Professor Crumbs demotes Alex and Justin to level one in the Family Wizard Competition, behind Max by two levels.
  Fact 0.10: In 'Alex Tells the World', Max sets traps in the Sub Station for the government but falls for one of them.
  Fact 0.11: Gregg Sulkin plays Mason as a special guest star in 'Alex Tells the World'.
  Fact 0.12: Ian Abercrombie guest stars as Professor Crumbs in 'Alex Tells the World'.
  Fact 0.13: Episode 81, 'Alex Gives Up', was directed by Victor Gonzalez and written by Todd J. Greenwald, and aired November 27, 2010 with production code 402.
  Fact 0.14: In 'Alex Gives Up', Alex decides to give up the wizard competition to spend more time with Mason.
  Fact 0.15: In 'Alex Gives Up', Alex thinks giving up does not matter since Max is all the way in the lead.
  Fact 0.16: In 'Alex Gives Up', Chancellor Tootietootie tells Alex that non-wizards and werewolves cannot date.
  Fact 0.17: In 'Alex Gives Up', Max's picture on the front of Future Wizards magazine attracts the daughter of a family of Cucuys.
  Fact 0.18: Cucuys are mythical Latino monsters.
  Fact 0.19: In 'Alex Gives Up', the Cucuys invite Max to a dinner party on their yacht.
  Fact 0.20: In 'Alex Gives Up', Alex plans to invite Chancellor Tootietootie to the Cucuy party to get him to appeal the law against werewolves and mortals dating.
  Fact 0.21: In 'Alex Gives Up', Lisa Cucuy finds out Mason is a werewolf and decides she wants him instead of Max.
  Fact 0.22: In 'Alex Gives Up', Alex gets jealous when Lisa flirts with Mason.
  Fact 0.23: In 'Alex Gives Up', Alex says embarrassing things about Mason to everyone on the yacht so he will get mad enough to turn into a werewolf.
  Fact 0.24: In 'Alex Gives Up', Carlos and Julie Cucuy jump off the boat because Cucuys are scared of werewolves.
  Fact 0.25: In 'Alex Gives Up', Jerry jumps off the boat after Carlos and Julie.
  Fact 0.26: In 'Alex Gives Up', Alex tells Lisa she can get off the boat the easy way or her way.
  Fact 0.27: In 'Alex Gives Up', Lisa jumps off the boat.
  Fact 0.28: In 'Alex Gives Up', Chancellor Tootietootie says the case will not be appealed because Mason turning into a werewolf proved the point that they are dangerous.
  Fact 0.29: In 'Alex Gives Up', Alex and Mason have to break up to be friends.
  Fact 0.30: In 'Alex Gives Up', Harper finishes writing her first novel based on the werewolf-wizard relationship.
  Fact 0.31: In 'Alex Gives Up', Jerry accidentally grabs Harper's just-finished novel while trying to get out of the water, causing him to fall back in.
  Fact 0.32: In 'Alex Gives Up', Justin tutors a class of delinquent wizards to gain a level to re-enter the Wizard Competition.
  Fact 0.33: Gregg Sulkin plays Mason as a special guest star in 'Alex Gives Up'.
  Fact 0.34: Andy Kindler guest stars as Chancellor Tootietootie in 'Alex Gives Up'.
  Fact 0.35: Frank Pacheco guest stars as Felix in 'Alex Gives Up'.
  Fact 0.36: David Barrera guest stars as Carlos Cucuy in 'Alex Gives Up'.
  Fact 0.37: Roxana Brusso guest stars as Julie Cucuy in 'Alex Gives Up'.
  Fact 0.38: Samantha Boscarino guest stars as Lisa Cucuy in 'Alex Gives Up'.
  Fact 0.39: Kari Wahlgren guest stars as Helen in 'Alex Gives Up'.
  Fact 0.40: Episode 82, 'Lucky Charmed', was directed by Victor Gonzalez and written by Justin Varava, and aired December 10, 2010 with production code 403 and 3.0 million U.S. viewers.
  Fact 0.41: In 'Lucky Charmed', Harper tries teaching Alex how to live without magic.
  Fact 0.42: In 'Lucky Charmed', Alex gets in big trouble for crashing Mr. Laritate's car, causing a disagreement with Harper.
  Fact 0.43: As punishment in 'Lucky Charmed', Alex has to clean the school's garbage cans.
  Fact 0.44: In 'Lucky Charmed', Mr. Laritate forgives Alex after seeing how the crash influenced her.
  Fact 0.45: In 'Lucky Charmed', Alex realizes she will be okay without magic.
  Fact 0.46: In 'Lucky Charmed', Alex uses Mr. Laritate's lucky life-saving hula girl figurine, Leilani, for luck.
  Fact 0.47: In 'Lucky Charmed', Alex turns into the hula girl and guides Harper through her driving test.
  Fact 0.48: In 'Lucky Charmed', Harper passes her driving test.
  Fact 0.49: In 'Lucky Charmed', Jerry wants to pass the wizard family robe to Max, but Justin has a hard time letting go of it.
  Fact 0.50: Bill Chott guest stars as Mr. Laritate in 'Lucky Charmed'.
  Fact 0.51: Julio Oscar Mechoso guest stars as DMV Instructor in 'Lucky Charmed'.
  Fact 0.52: Maria Canals Barrera, who plays Theresa Russo, was absent from 'Lucky Charmed'.
  Fact 0.53: Episode 83, 'Journey to the Center of Mason', was directed by Robbie Countryman and written by Peter Dirksen, and aired December 17, 2010 with production code 404.
  Fact 0.54: In 'Journey to the Center of Mason', Alex can no longer date Mason after finding out she will lose her powers.
  Fact 0.55: In 'Journey to the Center of Mason', Alex and Mason decide to try being best friends instead.
  Fact 0.56: In 'Journey to the Center of Mason', Dean, Alex's ex-boyfriend, asks Alex to get back together with him.
  Fact 0.57: In 'Journey to the Center of Mason', Mason gets jealous of Dean.
  Fact 0.58: In 'Journey to the Center of Mason', Max tells Mason that Dean and Alex used to date.
  Fact 0.59: In 'Journey to the Center of Mason', Mason turns into a werewolf and eats Dean.
  Fact 0.60: In 'Journey to the Center of Mason', Alex, Justin and Max turn Justin's toy Captain Jim Bob submarine big and go in it.
  Fact 0.61: In 'Journey to the Center of Mason', Alex, Justin and Max shrink the submarine and go into Mason's mouth.
  Fact 0.62: Inside Mason, Alex, Max and Justin see Mason's thoughts, which are all of Alex.
  Fact 0.63: In 'Journey to the Center of Mason', Alex, Max and Justin get Dean out.
  Fact 0.64: In 'Journey to the Center of Mason', Alex tells Dean he was knocked out and the doctor's solution was for them all to dress up as cows.
  Fact 0.65: In 'Journey to the Center of Mason', Dean offers again to get back together with Alex, and she says no.
  Fact 0.66: In 'Journey to the Center of Mason', Alex gets back together with Mason.
  Fact 0.67: Gregg Sulkin plays Mason as a special guest star in 'Journey to the Center of Mason'.
  Fact 0.68: Daniel Samonas guest stars as Dean in 'Journey to the Center of Mason'.
  Fact 0.69: Frank Pacheco guest stars as Felix in 'Journey to the Center of Mason'.
  Fact 0.70: Shoshana Bush guest stars as Tyler in 'Journey to the Center of Mason'.
  Fact 0.71: Episode 84, 'Three Maxes and a Little Lady', was directed by Victor Gonzalez and written by Richard Goodman, and aired January 7, 2011 with production code 405 and 4.0 million U.S. viewers.
  Fact 0.72: In 'Three Maxes and a Little Lady', Alex joins Justin's wizard class to try to move up in the competition.
  Fact 0.73: In 'Three Maxes and a Little Lady', Max joins a sophisticated wizard society.
  Fact 0.74: In 'Three Maxes and a Little Lady', the society suggests Max move the wizard competition up to a few days away.
  Fact 0.75: In 'Three Maxes and a Little Lady', Max quits the society because he finds out they are taking advantage of him.
  Fact 0.76: In 'Three Maxes and a Little Lady', Alex and Justin transform into Max to change the date of the family wizard competition back to its original date, and succeed.
  Fact 0.77: At the end of 'Three Maxes and a Little Lady', Alex and Justin accidentally turn Max indefinitely into a little girl.
  Fact 0.78: Bailee Madison guest stars as Maxine in 'Three Maxes and a Little Lady'.
  Fact 0.79: Frank Pacheco guest stars as Felix in 'Three Maxes and a Little Lady'.
  Fact 0.80: Kari Wahlgren guest stars as Helen in 'Three Maxes and a Little Lady'.
  Fact 0.81: Shane Harper guest stars as Fidel in 'Three Maxes and a Little Lady'.
  Fact 0.82: Bailee Madison begins her recurring role as Maxine, the female Max, in 'Three Maxes and a Little Lady'.
  Fact 0.83: Jake T. Austin was absent in four other episodes and returns at the end of the tenth episode, 'Back to Max'.
  Fact 0.84: Episode 85, 'Daddy's Little Girl', was directed by Robbie Countryman and written by Gigi McCreery and Perry Rein, and aired January 21, 2011 with production code 406 and 3.4 million U.S. viewers.
  Fact 0.85: In 'Daddy's Little Girl', the parents pay more attention to Max, who is now Maxine, and Alex becomes jealous.
  Fact 0.86: In 'Daddy's Little Girl', Justin takes Maxine to her karate class, where Maxine kicks Justin's butt.
  Fact 0.87: In 'Daddy's Little Girl', Alex makes Jerry's favorite sandwich to win his love back.
  Fact 0.88: In 'Daddy's Little Girl', Maxine announces she beat Justin in karate, and Jerry falls for her cuteness again.
  Fact 0.89: In 'Daddy's Little Girl', Alex and Justin cast spells on Maxine to change her back, which only makes her cuter and furious.
  Fact 0.90: In 'Daddy's Little Girl', Maxine demands she kicks either Alex's or Justin's butt.
  Fact 0.91: In 'Daddy's Little Girl', Alex casts a spell on Justin to make him a karate master.
  Fact 0.92: In 'Daddy's Little Girl', Justin beats Maxine, and Maxine is hurt physically.
  Fact 0.93: In 'Daddy's Little Girl', Jerry and Theresa rush to Maxine's side.
  Fact 0.94: In 'Daddy's Little Girl', Justin leaves, and Alex leaves after him, heartbroken.
  Fact 0.95: In 'Daddy's Little Girl', Jerry tells Alex he fell for Maxine because she reminded him of Alex when she was little.
  Fact 0.96: In 'Daddy's Little Girl', Zeke thinks the Russos hate him, but it was really Harper's alibi to keep him away from the Sub Station.
  Fact 0.97: Bailee Madison guest stars as Maxine in 'Daddy's Little Girl'.
  Fact 0.98: Dan Benson guest stars as Zeke in 'Daddy's Little Girl'.
  Fact 0.99: Lanny Joon guest stars as Sensei in 'Daddy's Little Girl'.
  Fact 0.100: Jake T. Austin, who plays Max, was absent from 'Daddy's Little Girl'.
  Fact 0.101: Episode 86, 'Everything's Rosie for Justin', was directed by Victor Gonzalez and written by Justin Varava, and aired February 4, 2011 with production code 407 and 3.9 million U.S. viewers.
  Fact 0.102: In 'Everything's Rosie for Justin', Justin and his delinquent class have a wizard wand dance assessment to get them back in Wiztech.
  Fact 0.103: In 'Everything's Rosie for Justin', Rosie, a new student, is an angel.
  Fact 0.104: In 'Everything's Rosie for Justin', Justin falls madly in love with Rosie.
  Fact 0.105: In 'Everything's Rosie for Justin', Harper makes cards that customers get hole punched each time they buy a sandwich, and nine punches earn a free sandwich.
  Fact 0.106: In 'Everything's Rosie for Justin', one man punches his own card, demands a free sandwich, and brings dozens of people with pre-punched cards.
  Fact 0.107: In 'Everything's Rosie for Justin', Maxine uses her cuteness to convince customers not to exploit her parents.
  Fact 0.108: In 'Everything's Rosie for Justin', Justin kicks Rosie out of his class so it does not fail the assessment.
  Fact 0.109: In 'Everything's Rosie for Justin', Alex gets Rosie to rejoin the class and puts a copycat spell on her.
  Fact 0.110: In 'Everything's Rosie for Justin', a squirrel-frog attaches itself to Alex's foot during the assessment.
  Fact 0.111: In 'Everything's Rosie for Justin', Rosie reveals herself to be an angel during the assessment.
  Fact 0.112: In 'Everything's Rosie for Justin', the class fails the assessment.
  Fact 0.113: In 'Everything's Rosie for Justin', Justin starts dating Rosie.
  Fact 0.114: Bailee Madison guest stars as Maxine in 'Everything's Rosie for Justin'.
  Fact 0.115: Frank Pacheco guest stars as Felix in 'Everything's Rosie for Justin'.
  Fact 0.116: Leven Rambin guest stars as Rosie in 'Everything's Rosie for Justin'.
  Fact 0.117: Diane Delano guest stars as Penny in 'Everything's Rosie for Justin'.
  Fact 0.118: Chris Coppola guest stars as Customer in 'Everything's Rosie for Justin'.
  Fact 0.119: Cameron Sanders guest stars as Nelvis in 'Everything's Rosie for Justin'.
  Fact 0.120: Jake T. Austin, who plays Max, was absent from 'Everything's Rosie for Justin'.
  Fact 0.121: Episode 87, 'Dancing with Angels', was directed by Victor Gonzalez and written by Richard Goodman, and aired February 11, 2011 with production code 408 and 4.5 million U.S. viewers.
  Fact 0.122: In 'Dancing with Angels', Justin and Rosie start dating, but their date gets ruined, so they want a second date.
  Fact 0.123: In 'Dancing with Angels', Alex suggests an Angel Club in Los Angeles, but Rosie says only Angels are allowed.
  Fact 0.124: In 'Dancing with Angels', Maxine will not tell their parents as long as Alex brings her back an Ozzy Osbourne stone.
  Fact 0.125: In 'Dancing with Angels', Justin, Harper and Alex sneak into the Angel Club with Rosie.
  Fact 0.126: In 'Dancing with Angels', Zeedrik asks if they are angels, and they say they are.
  Fact 0.127: In 'Dancing with Angels', Theresa and Jerry force Maxine to join a pageant so she will tell them where the others are.
  Fact 0.128: In 'Dancing with Angels', Alex and Harper cannot fly, so Justin secretly uses his wand to give Harper flying powers.
  Fact 0.129: In 'Dancing with Angels', Zeedrik tests them by having them sing while playing the harp, and they fail.
  Fact 0.130: In 'Dancing with Angels', Alex, Justin and Harper are accepted as angels until the key-chain of Ozzy Osbourne's star reverts to normal.
  Fact 0.131: In 'Dancing with Angels', Zeedrik believes they are really angels of darkness, and everyone in the club runs off, even Rosie.
  Fact 0.132: In 'Dancing with Angels', Rosie comes back to comfort Justin, and her wings are shown to be black.
  Fact 0.133: In 'Dancing with Angels', Justin does not see Rosie's black wings because he is looking forward.
  Fact 0.134: In 'Dancing with Angels', Maxine wins the beauty pageant, and Jerry and Theresa celebrate.
  Fact 0.135: Bailee Madison guest stars as Maxine in 'Dancing with Angels'.
  Fact 0.136: Leven Rambin guest stars as Rosie in 'Dancing with Angels'.
  Fact 0.137: Ransford Doherty guest stars as Zeedrik in 'Dancing with Angels'.
  Fact 0.138: Jake T. Austin, who plays Max, was absent from 'Dancing with Angels'.
  Fact 0.139: Episode 88, 'Wizards vs. Angels', was a one-hour special directed by Victor Gonzalez and written by Vince Cheung and Ben Montanio, and aired February 18, 2011 with production codes 409 and 410 and 5.1 million U.S. viewers.
  Fact 0.140: In 'Wizards vs. Angels', Rosie is an Angel of Darkness and has been tricking Justin to make him evil.
  Fact 0.141: In 'Wizards vs. Angels', Alex finds out Rosie is an Angel of Darkness with the help of Tina, a guardian angel in training.
  Fact 0.142: In 'Wizards vs. Angels', Justin crosses over to the dark side with Rosie.
  Fact 0.143: In 'Wizards vs. Angels', Justin steals the moral compass and gives it to Gorog so Gorog can turn it from good to bad.
  Fact 0.144: In 'Wizards vs. Angels', Rosie tries to get Justin to run away because Gorog wants to destroy him.
  Fact 0.145: In 'Wizards vs. Angels', Justin refuses to leave and breaks his wand in halves.
  Fact 0.146: In 'Wizards vs. Angels', Alex tries to convince Justin that he is a good wizard.
  Fact 0.147: In 'Wizards vs. Angels', Maxine is enrolled in Tribeca Prep as part of a cousin exchange program.
  Fact 0.148: In 'Wizards vs. Angels', Harper insists Maxine have a slumber party, which becomes a disaster.
  Fact 0.149: In 'Wizards vs. Angels', Tina reveals that Rosie was a good angel, and Gorog corrupted her before her transport arrived.
  Fact 0.150: In 'Wizards vs. Angels', Alex and Tina go to the Dark Realm to save Justin.
  Fact 0.151: In 'Wizards vs. Angels', Alex borrows a pair of wings from the guardian angels.
  Fact 0.152: In 'Wizards vs. Angels', Alex takes the Moral Compass and tries to turn it to good, but Justin stops her.
  Fact 0.153: In 'Wizards vs. Angels', Rosie uses her good angel side, and they fly up.
  Fact 0.154: In 'Wizards vs. Angels', Justin turns the compass to good when they land.
  Fact 0.155: In 'Wizards vs. Angels', Tina and Alex take the compass and run.
  Fact 0.156: In 'Wizards vs. Angels', when Rosie and Justin kiss, Rosie turns into a good angel with white wings.
  Fact 0.157: In 'Wizards vs. Angels', Justin loses his dark wings.
  Fact 0.158: Bill Chott guest stars as Mr. Laritate in 'Wizards vs. Angels'.
  Fact 0.159: Bailee Madison guest stars as Maxine in 'Wizards vs. Angels'.
  Fact 0.160: Leven Rambin guest stars as Rosie in 'Wizards vs. Angels'.
  Fact 0.161: John Rubinstein guest stars as Gorog in 'Wizards vs. Angels'.
  Fact 0.162: China McClain guest stars as Tina in 'Wizards vs. Angels'.
  Fact 0.163: Amy Tolsky guest stars as Florence in 'Wizards vs. Angels'.
  Fact 0.164: Jake T. Austin, who plays Max, was absent from 'Wizards vs. Angels'.
  Fact 0.165: Episode 89, 'Back To Max', was directed by Guy Distad and written by Todd J. Greenwald, and aired March 11, 2011 with production code 411 and 3.7 million U.S. viewers.
  Fact 0.166: In 'Back To Max', Justin invites Professor Crumbs to check on his delinquent wizards class.
  Fact 0.167: In 'Back To Max', Maxine interrupts, and Alex and Justin try to find a way to turn Maxine back to Max.
  Fact 0.168: In 'Back To Max', Harper takes a role in the school's Spirit of America play, but is replaced by Maxine.
  Fact 0.169: Jake T. Austin returns as Max near the end of 'Back To Max'.
  Fact 0.170: Bailee Madison marks her final appearance as Maxine in 'Back To Max'.
  Fact 0.171: Bill Chott guest stars as Mr. Laritate in 'Back To Max'.
  Fact 0.172: Ian Abercrombie guest stars as Professor Crumbs in 'Back To Max'.
  Fact 0.173: Bailee Madison guest stars as Maxine in 'Back To Max'.
  Fact 0.174: Cameron Sanders guest stars as Nelvis in 'Back To Max'.
  Fact 0.175: LJ Benet guest stars as Little Crumbs in 'Back To Max'.
  Fact 0.176: Jackie Evancho guest stars as Choir Girl in 'Back To Max'.
  Fact 0.177: Episode 90, 'Zeke Finds Out', was directed by David DeLuise and written by Peter Dirksen, and aired April 8, 2011 with production code 412 and 3.0 million U.S. viewers.
  Fact 0.178: In 'Zeke Finds Out', Alex uses magic to make Zeke's tricks come true after she gets bored of his failing magic tricks.
  Fact 0.179: In 'Zeke Finds Out', Zeke tells Harper he believes he is a wizard and books a magic act at the sub station for a kid's birthday party.
  Fact 0.180: In 'Zeke Finds Out', Justin and Harper decide to leave everything up to Alex.
  Fact 0.181: In 'Zeke Finds Out', Harper tells Alex to tell Zeke the truth about wizards.
  Fact 0.182: In 'Zeke Finds Out', Alex is reluctant to expose wizardry because it could get her kicked out of the wizard competition forever.
  Fact 0.183: In 'Zeke Finds Out', Zeke breaks up with Harper after finding out she is keeping a secret from him.
  Fact 0.184: In 'Zeke Finds Out', Alex makes Harper disappear before she can expose wizardry to Zeke.
  Fact 0.185: In 'Zeke Finds Out', Justin freezes everyone except himself and Zeke.
  Fact 0.186: In 'Zeke Finds Out', Justin tells Zeke that he is a wizard and flashes him into their lair to unveil the secret.
  Fact 0.187: In 'Zeke Finds Out', Zeke promises he will never tell anyone about wizards.
  Fact 0.188: In 'Zeke Finds Out', Zeke and Harper reconcile, and Alex and Harper apologize and make up.
  Fact 0.189: Dan Benson guest stars as Zeke in 'Zeke Finds Out'.
  Fact 0.190: Maurice G. Smith guest stars as Big Mitch in 'Zeke Finds Out'.
  Fact 0.191: Maria Canals Barrera, who plays Theresa Russo, was absent from 'Zeke Finds Out'.
  Fact 0.192: David DeLuise, who plays Jerry Russo, was absent from 'Zeke Finds Out'.
  Fact 0.193: Episode 91, 'Magic Unmasked', was directed by David DeLuise and written by Justin Varava, and aired May 13, 2011 with production code 413 and 2.7 million U.S. viewers.
  Fact 0.194: In 'Magic Unmasked', a former legendary Luchador named Muy Macho visits the Sub Station.
  Fact 0.195: In 'Magic Unmasked', Alex finds out she was the reason Muy Macho quit lucha libre.
  Fact 0.196: In 'Magic Unmasked', Alex arranges a lucha libre match with Muy Macho and Jerry to restore his honor.
  Fact 0.197: In 'Magic Unmasked', Zeke asks too many questions about wizardry.
  Fact 0.198: In 'Magic Unmasked', Justin offers Zeke one magical wish if he promises to stop questioning him about magic.
  Fact 0.199: In 'Magic Unmasked', Zeke's wish is to defeat Muy Macho in the match.
  Fact 0.200: In 'Magic Unmasked', Max pretends to like Talia's hobbies but does not enjoy them.
  Fact 0.201: Dan Benson guest stars as Zeke in 'Magic Unmasked'.
  Fact 0.202: Valente Rodriguez guest stars as Muy Macho in 'Magic Unmasked'.
  Fact 0.203: McKaley Miller guest stars as Talia in 'Magic Unmasked'.
  Fact 0.204: Episode 92, 'Meet the Werewolves', was directed by Victor Gonzalez and written by David Henrie, and aired June 17, 2011 with production code 415 and 4.0 million U.S. viewers.
  Fact 0.205: In 'Meet the Werewolves', Alex asks to meet Mason's parents, and Mason shows her fake parents.
  Fact 0.206: In 'Meet the Werewolves', Mason takes Alex to the Autumn Moon Feast, a werewolf holiday, to meet his real parents.
  Fact 0.207: In 'Meet the Werewolves', Alex uses a spell to change herself and Harper into werewolves.
  Fact 0.208: In 'Meet the Werewolves', Alex tries to be the worst werewolf possible in front of Mason's parents.
  Fact 0.209: In 'Meet the Werewolves', Mason's parents like Alex as a werewolf.
  Fact 0.210: In 'Meet the Werewolves', Mason tells his parents that he truly loves Alex and that she is a wizard.
  Fact 0.211: In 'Meet the Werewolves', Alex changes back to a wizard and Mason turns back to human form.
  Fact 0.212: In 'Meet the Werewolves', Mason's parents try to eat Alex and Mason, and they make a near escape.
  Fact 0.213: In 'Meet the Werewolves', Max makes his parents' food taste like kid food.
  Fact 0.214: In 'Meet the Werewolves', Jerry and Theresa act like little kids after eating the food.
  Fact 0.215: In 'Meet the Werewolves', Justin uses a spell on a pizza to turn Jerry and Theresa back to their regular selves.
  Fact 0.216: In 'Meet the Werewolves', Max uses a spell on cookies to make them older, and they turn into teenagers.
  Fact 0.217: Gregg Sulkin plays Mason as a special guest star in 'Meet the Werewolves'.
  Fact 0.218: Harry Van Gorkum guest stars as Grant Greybeck in 'Meet the Werewolves'.
  Fact 0.219: Robin Riker guest stars as Linda Greybeck in 'Meet the Werewolves'.
  Fact 0.220: Marianne Muellerleile guest stars as Molly in 'Meet the Werewolves'.
  Fact 0.221: Eric Zuckerman guest stars as Bill in 'Meet the Werewolves'.
  Fact 0.222: Episode 93, 'Beast Tamer', was directed by Victor Gonzalez and written by Gigi McCreery and Perry Rein, and aired June 24, 2011 with production code 416 and 4.0 million U.S. viewers.
  Fact 0.223: In 'Beast Tamer', Alex, Justin and Max win four tickets to the Beast Bowl.
  Fact 0.224: In 'Beast Tamer', Beast Tamer Chase Riprock delivers the tickets, flirts with Alex, and invites the group on a pre-bowl tour.
  Fact 0.225: In 'Beast Tamer', Alex invites Mason to go with them, but he says he cannot.
  Fact 0.226: In 'Beast Tamer', Justin tries to tame the beast so Max can have his photo taken with it, and both are caught and banned from the event.
  Fact 0.227: In 'Beast Tamer', Chase tries to kiss Alex, but she rejects him and returns to the mortal world.
  Fact 0.228: In 'Beast Tamer', Alex finds Mason coming out of a hardware store, where he is evasive about his plans.
  Fact 0.229: In 'Beast Tamer', Mason is making an anniversary present for Alex, which is a sculpture of himself and Alex.
  Fact 0.230: In 'Beast Tamer', Harper tells Mason that Chase likes Alex and urges him to surprise her with the gift at the Beast Bowl.
  Fact 0.231: In 'Beast Tamer', Justin and Max return to the Beast Bowl disguised as beast taming clowns.
  Fact 0.232: In 'Beast Tamer', the beast escapes and snatches Alex, and Alex throws the whip to Chase.
  Fact 0.233: In 'Beast Tamer', Chase tames the beast, which drops Alex.
  Fact 0.234: In 'Beast Tamer', Mason reveals his gift to Alex.
  Fact 0.235: In 'Beast Tamer', Alex and Mason leave together with the sculpture.
  Fact 0.236: Gregg Sulkin plays Mason as a special guest star in 'Beast Tamer'.
  Fact 0.237: Nick Roux guest stars as Chase in 'Beast Tamer'.
  Fact 0.238: Maria Canals Barrera, who plays Theresa Russo, was absent from 'Beast Tamer'.
  Fact 0.239: David DeLuise, who plays Jerry Russo, was absent from 'Beast Tamer'.
  Fact 0.240: Episode 94, 'Wizard of the Year', was directed by Victor Gonzalez and written by Richard Goodman, and aired July 8, 2011 with production code 417 and 3.6 million U.S. viewers.
  Fact 0.241: In 'Wizard of the Year', Alex is crowned Wizard of the Year for saving the world from dark angels.
  Fact 0.242: In 'Wizard of the Year', Alex is reinstated in the Wizard Competition.
  Fact 0.243: In 'Wizard of the Year', a banquet honors Alex's achievement.
  Fact 0.244: In 'Wizard of the Year', Chase Riprock visits the lair to congratulate Alex.
  Fact 0.245: In 'Wizard of the Year', Keith Keith hounds them about being in a romantic relationship, and Alex says she already has a boyfriend.
  Fact 0.246: In 'Wizard of the Year', Keith Keith confirms that Chase and Alex are dating.
  Fact 0.247: In 'Wizard of the Year', Justin's video only contains him complaining about Alex.
  Fact 0.248: In 'Wizard of the Year', Mason stands Alex up on the night of the awards banquet.
  Fact 0.249: In 'Wizard of the Year', Chancellor TootieTootie tells Justin his display of affection and pride for Alex may reinstate him into the competition.
  Fact 0.250: In 'Wizard of the Year', Justin tries to destroy the tape.
  Fact 0.251: In 'Wizard of the Year', Chase sits at the Russos' table at the banquet.
  Fact 0.252: In 'Wizard of the Year', Mason sees Alex and Chase flirting and a short battle between Mason and Chase ensues.
  Fact 0.253: In 'Wizard of the Year', Alex breaks up the fight and chooses neither Mason nor Chase.
  Fact 0.254: In 'Wizard of the Year', Alex breaks up with Mason.
  Fact 0.255: Gregg Sulkin plays Mason as a special guest star in 'Wizard of the Year'.
  Fact 0.256: Andy Kindler guest stars as Chancellor Tootietootie in 'Wizard of the Year'.
  Fact 0.257: Nick Roux guest stars as Chase in 'Wizard of the Year'.
  Fact 0.258: Justin Guarini guest stars as Keith Keith in 'Wizard of the Year'.
  Fact 0.259: Episode 95, 'Misfortune at the Beach', was directed by Jody Margolin Hahn and written by Vince Cheung and Ben Montanio, and aired July 24, 2011 with production code 420 and 3.5 million U.S. viewers.
  Fact 0.260: In 'Misfortune at the Beach', the Russos go to the beach on an extremely hot day.
  Fact 0.261: In 'Misfortune at the Beach', the Russos come across Zelzar, a fortune-telling machine.
  Fact 0.262: In 'Misfortune at the Beach', Jerry warns the kids that Zelzar is from the wizard world and a wizard's fortune really comes true.
  Fact 0.263: In 'Misfortune at the Beach', Alex gets a fortune that says 'Say good-bye to your life.'
  Fact 0.264: In 'Misfortune at the Beach', Alex comes close to life-threatening situations multiple times.
  Fact 0.265: In 'Misfortune at the Beach', Alex, Justin, Max and Harper make a compromise with Zelzar: if they extract him from the machine and let him enjoy a day at the beach, he will take back Alex's fortune.
  Fact 0.266: In 'Misfortune at the Beach', Zelzar reveals he cannot take away Alex's fortune but can give it to the next person.
  Fact 0.267: In 'Misfortune at the Beach', a little girl gets Alex's fortune.
  Fact 0.268: In 'Misfortune at the Beach', a man from Random Prize Giveaway says 'Say good-bye to your life because we're randomly giving you one million dollars!'
  Fact 0.269: Michael Carbonaro guest stars as Zelzar in 'Misfortune at the Beach'.
  Fact 0.270: John J. York guest stars as Game Show Host in 'Misfortune at the Beach'.
  Fact 0.271: Episode 96, 'Wizards vs. Asteroid', was directed by Victor Gonzalez and written by Peter Dirksen, and aired August 19, 2011 with production code 418 and 3.6 million U.S. viewers.
  Fact 0.272: In 'Wizards vs. Asteroid', a huge asteroid is hurtling towards the earth.
  Fact 0.273: In 'Wizards vs. Asteroid', the Russos decide to hide in the wizard world, but Alex, Justin and Max go into space to destroy the asteroid.
  Fact 0.274: In 'Wizards vs. Asteroid', Alex finds out she is not graduating high school.
  Fact 0.275: In 'Wizards vs. Asteroid', a small part of the asteroid hits the school, giving Alex a chance to graduate.
  Fact 0.276: Dan Benson guest stars as Zeke in 'Wizards vs. Asteroid'.
  Fact 0.277: Bill Chott guest stars as Mr. Laritate in 'Wizards vs. Asteroid'.
  Fact 0.278: Kristina Hayes guest stars as Reporter in 'Wizards vs. Asteroid'.
  Fact 0.279: Episode 97, 'Justin's Back In', was directed by Guy Distad and written by Todd J. Greenwald, and aired August 26, 2011 with production code 419 and 3.2 million U.S. viewers.
  Fact 0.280: In 'Justin's Back In', Justin's class of delinquent wizards has completed their studies and can rejoin WizTech if they all pass their final evaluation.
  Fact 0.281: In 'Justin's Back In', all of the students fail the final evaluation, which Alex suspects is wrong.
  Fact 0.282: In 'Justin's Back In', the class had actually passed, but the historian who checked the exams was evil and lied about the scores.
  Fact 0.283: In 'Justin's Back In', the evil historian attacks Justin's class, and Justin stops him and saves Alex.
  Fact 0.284: In 'Justin's Back In', a student stops the historian by pulling the most powerful wand in the wizard world out of a crystal ball.
  Fact 0.285: In 'Justin's Back In', the student is a descendant of a famous wizard who was previously thought to have no living descendants.
  Fact 0.286: In 'Justin's Back In', Justin's class is readmitted into WizTech, and Justin is officially back in the family wizard competition.
  Fact 0.287: In 'Justin's Back In', Jerry and Theresa try to recreate Max's childhood accomplishments with the help of Harper.
  Fact 0.288: In 'Justin's Back In', during the credits it is revealed that Max had kept his favorite memories all along.
  Fact 0.289: Tim Conway guest stars as Cragmont, as special guest star, in 'Justin's Back In'.
  Fact 0.290: Ian Abercrombie guest stars as Professor Crumbs in 'Justin's Back In'.
  Fact 0.291: Frank Pacheco guest stars as Felix in 'Justin's Back In'.
  Fact 0.292: Cameron Sanders guest stars as Nelvis in 'Justin's Back In'.
  Fact 0.293: Episode 98, 'Alex the Puppetmaster', was directed by Victor Gonzalez and written by Gigi McCreery and Perry Rein, and aired September 16, 2011 with production code 421 and 3.0 million U.S. viewers.
  Fact 0.294: In 'Alex the Puppetmaster', Alex and Harper decide to do a puppet show to raise money for a new apartment, and Zeke and Justin help them.
  Fact 0.295: In 'Alex the Puppetmaster', Alex forgets to write the lines for the show.
  Fact 0.296: In 'Alex the Puppetmaster', Harper gets mad and does a show of her own, with puppets that act like the way Alex treats her.
  Fact 0.297: In 'Alex the Puppetmaster', Alex makes Justin and Zeke puppets for her show and steals Harper's crowd.
  Fact 0.298: In 'Alex the Puppetmaster', Alex apologizes to Harper using the puppets.
  Fact 0.299: In 'Alex the Puppetmaster', Talia's parents meet Theresa and Jerry for the first time, and it does not go well.
  Fact 0.300: In 'Alex the Puppetmaster', Justin puts a spell on Talia's parents to forget they ever met Theresa and Jerry.
  Fact 0.301: In 'Alex the Puppetmaster', Talia's parents meet Theresa and Jerry again and end up watching a 16-hour play together.
  Fact 0.302: Joely Fisher guest stars as Meg Robinson, as special guest star, in 'Alex the Puppetmaster'.
  Fact 0.303: Dan Benson guest stars as Zeke in 'Alex the Puppetmaster'.
  Fact 0.304: James Urbaniak guest stars as Rob Robinson in 'Alex the Puppetmaster'.
  Fact 0.305: McKaley Miller guest stars as Talia in 'Alex the Puppetmaster'.
  Fact 0.306: Episode 99, 'My Two Harpers', was directed by Victor Gonzalez and written by Manny Basanese, and aired September 30, 2011 with production code 422 and 3.0 million U.S. viewers.
  Fact 0.307: In 'My Two Harpers', Harper and Zeke spend so much time together that Alex decides to make a Harper clone.
  Fact 0.308: In 'My Two Harpers', Justin comes up with ideas to save money in the sub-shop.
  Fact 0.309: In 'My Two Harpers', Max starts a fast-food restaurant in the wizard world.
  Fact 0.310: Dan Benson guest stars as Zeke in 'My Two Harpers'.
  Fact 0.311: Maria Canals Barrera, who plays Theresa Russo, was absent from 'My Two Harpers'.
  Fact 0.312: Episode 100, 'Wizards of Apartment 13B', was directed by Guy Distad and written by Justin Varava, and aired October 7, 2011 with production code 423 and 3.5 million U.S. viewers.
  Fact 0.313: In 'Wizards of Apartment 13B', Alex and Harper receive a flyer for a hotel.
  Fact 0.314: In 'Wizards of Apartment 13B', the hotel's owner, Dexter, sells them Apartment 13B, which belongs to a floor for non-humans.
  Fact 0.315: In 'Wizards of Apartment 13B', Harper is forced to use a training wand that can only open things.
  Fact 0.316: In 'Wizards of Apartment 13B', Mason lives on the floor and will not stop trying to win Alex back.
  Fact 0.317: In 'Wizards of Apartment 13B', Alex and Harper throw a party that the whole floor attends, including Felix.
  Fact 0.318: In 'Wizards of Apartment 13B', Mason uses Harper's wand to open Alex's heart, but she flirts with an ugly giant.
  Fact 0.319: In 'Wizards of Apartment 13B', Mason makes Alex open her heart to Felix.
  Fact 0.320: In 'Wizards of Apartment 13B', Theresa and Jerry trick Justin into going to Alex's apartment.
  Fact 0.321: In 'Wizards of Apartment 13B', Justin fixes Mason's mistakes by resetting Harper's wand, and it is revealed that he misses Alex.
  Fact 0.322: In 'Wizards of Apartment 13B', Alex, freed from Mason's spell, is angry at Mason, and Mason leaves.
  Fact 0.323: In 'Wizards of Apartment 13B', Dexter turns into Gorog outside the door when no one is looking.
  Fact 0.324: In 'Wizards of Apartment 13B', Max appears in the apartment, and Alex and Harper kick him out.
  Fact 0.325: In 'Wizards of Apartment 13B', Gorog gives Max a flyer for a free camp, and Max accepts.
  Fact 0.326: 'Wizards of Apartment 13B' is part 1 of a 4-part arc.
  Fact 0.327: 'Wizards of Apartment 13B' is the 100th episode.
  Fact 0.328: Gregg Sulkin plays Mason as a special guest star in 'Wizards of Apartment 13B'.
  Fact 0.329: John Rubinstein guest stars as Gorog in 'Wizards of Apartment 13B'.
  Fact 0.330: Frank Pacheco guest stars as Felix in 'Wizards of Apartment 13B'.
  Fact 0.331: Fred Stoller guest stars as Dexter in 'Wizards of Apartment 13B'.
  Fact 0.332: Christopher Douglas Reed guest stars as Ogre in 'Wizards of Apartment 13B'.
  Fact 0.333: Episode 101, 'Ghost Roommate', was directed by David DeLuise and written by Richard Goodman, and aired October 14, 2011 with production code 424 and 4.1 million U.S. viewers.
  Fact 0.334: In 'Ghost Roommate', Alex and Harper have trouble paying the apartment bills, and Dexter recommends getting a roommate.
  Fact 0.335: In 'Ghost Roommate', Alex and Harper find a wealthy ghost named Lucy and let her move in.
  Fact 0.336: In 'Ghost Roommate', Justin orders a robot to help Zeke at the sub station.
  Fact 0.337: In 'Ghost Roommate', Dexter, who is Gorog in disguise, makes an evil robot to spy on Justin and pass information to Gorog.
  Fact 0.338: In 'Ghost Roommate', Lucy explains that her boyfriend Donny disappeared sixty years prior in a plane accident.
  Fact 0.339: In 'Ghost Roommate', Alex tries to set Lucy up with Justin, but he makes matters worse.
  Fact 0.340: In 'Ghost Roommate', Lucy goes on a long date with Mason, which makes Alex jealous.
  Fact 0.341: In 'Ghost Roommate', Alex helps Lucy locate Donny, who had been stranded on an island in the Bermuda Triangle.
  Fact 0.342: In 'Ghost Roommate', Alex realizes she cannot get back home because heavy magnetic currents interfere with her magic.
  Fact 0.343: In 'Ghost Roommate', Mason goes to save Alex after he finds out she left for the Bermuda Triangle.
  Fact 0.344: In 'Ghost Roommate', the robot attacks Justin and Zeke after they find it scanning the spell book.
  Fact 0.345: In 'Ghost Roommate', the robot gives Gorog the spell he needed that he got from the Russos.
  Fact 0.346: In 'Ghost Roommate', Mason and Alex get off the island using the Bermuda shorts given to them by Justin.
  Fact 0.347: Gregg Sulkin plays Mason as a special guest star in 'Ghost Roommate'.
  Fact 0.348: Dan Benson guest stars as Zeke in 'Ghost Roommate'.
  Fact 0.349: Fred Stoller guest stars as Dexter in 'Ghost Roommate'.
  Fact 0.350: Josh Sussman guest stars as Hugh Normous in 'Ghost Roommate'.
  Fact 0.351: Linsey Godfrey guest stars as Lucy in 'Ghost Roommate'.
  Fact 0.352: Travis Caldwell guest stars as Donny in 'Ghost Roommate'.
  Fact 0.353: Michael Carbonaro guest stars as Robot in 'Ghost Roommate'.
  Fact 0.354: 'Ghost Roommate' is part 2 of a 4-part arc.
  Fact 0.355: Jake T. Austin, Maria Canals Barrera and David DeLuise were absent from 'Ghost Roommate'.
  Fact 0.356: Episode 102, 'Get Along, Little Zombie', was directed by Victor Gonzalez and written by Peter Dirksen, and aired October 21, 2011 with production code 425 and 3.4 million U.S. viewers.
  Fact 0.357: In 'Get Along, Little Zombie', Alex and Harper run into Mr. Laritate, who also lives in the hotel.
  Fact 0.358: In 'Get Along, Little Zombie', Felix tells Justin he broke his wand, and Justin says it is fake because he saw batteries in it.
  Fact 0.359: In 'Get Along, Little Zombie', Justin concludes that someone stole Felix's wand.
  Fact 0.360: In 'Get Along, Little Zombie', Dexter shows a zombie to the hall for magical creatures.
  Fact 0.361: In 'Get Along, Little Zombie', Mason messes with the elevator buttons so he can spend more time with Alex, and they land on the second floor.
  Fact 0.362: In 'Get Along, Little Zombie', all the humans discover the mysterious thirteenth floor.
  Fact 0.363: In 'Get Along, Little Zombie', Alex lies that they accidentally got into a secret open house floor.
  Fact 0.364: In 'Get Along, Little Zombie', Mr. Laritate discovers the thirteenth floor and sees Alex, Harper and Mason.
  Fact 0.365: In 'Get Along, Little Zombie', the zombie bites Mr. Laritate and turns him into a zombie.
  Fact 0.366: In 'Get Along, Little Zombie', Alex looks for a spell to turn Mr. Laritate human again.
  Fact 0.367: In 'Get Along, Little Zombie', Mr. Laritate runs away to the substation.
  Fact 0.368: In 'Get Along, Little Zombie', Professor Crumbs questions Felix, who reveals the 13th floor, but Crumbs denies that a 13th floor exists.
  Fact 0.369: In 'Get Along, Little Zombie', Alex and Mason are forced into a dance while trying to sneak the zombie into the substation.
  Fact 0.370: In 'Get Along, Little Zombie', Harper causes a distraction, and they give Mr. Laritate a medicine from the wizard world that turns him human.
  Fact 0.371: In 'Get Along, Little Zombie', Mr. Laritate remembers nothing.
  Fact 0.372: In 'Get Along, Little Zombie', Alex kisses Mr. Laritate on the cheek, and Mason suggests they just be friends.
  Fact 0.373: In 'Get Along, Little Zombie', Alex gets back together with Mason.
  Fact 0.374: In 'Get Along, Little Zombie', Crumbs tells Felix the floor should not exist.
  Fact 0.375: In 'Get Along, Little Zombie', Justin tells them the floor does not exist.
  Fact 0.376: In 'Get Along, Little Zombie', Crumbs thinks there is evil going on.
  Fact 0.377: In 'Get Along, Little Zombie', Dexter comes with dark angel wings and is revealed to have stolen Felix's wand.
  Fact 0.378: In 'Get Along, Little Zombie', Dexter tricks Felix into turning to the dark side in return for the wand.
  Fact 0.379: In 'Get Along, Little Zombie', Felix casts a spell that traps them all on the floor.
  Fact 0.380: In 'Get Along, Little Zombie', Mr. Laritate is dancing back at Waverly Place while Jerry sells chili.
  Fact 0.381: In 'Get Along, Little Zombie', Mr. Laritate eats chili and gets water to get the burn out of his mouth.
  Fact 0.382: Gregg Sulkin plays Mason as a special guest star in 'Get Along, Little Zombie'.
  Fact 0.383: Bill Chott guest stars as Mr. Laritate in 'Get Along, Little Zombie'.
  Fact 0.384: Ian Abercrombie guest stars as Professor Crumbs in 'Get Along, Little Zombie'.
  Fact 0.385: Frank Pacheco guest stars as Felix in 'Get Along, Little Zombie'.
  Fact 0.386: Fred Stoller guest stars as Dexter in 'Get Along, Little Zombie'.
  Fact 0.387: Regan Burns guest stars as Everett in 'Get Along, Little Zombie'.
  Fact 0.388: Jane Carr guest stars as Martha St. Claire in 'Get Along, Little Zombie'.
  Fact 0.389: 'Get Along, Little Zombie' is part 3 of a 4-part arc.
  Fact 0.390: Jake T. Austin and Maria Canals Barrera were absent from 'Get Along, Little Zombie'.
  Fact 0.391: Episode 103, 'Wizards vs. Everything', was directed by Victor Gonzalez and written by Gigi McCreery and Perry Rein, and aired October 28, 2011 with production code 426 and 3.4 million U.S. viewers.
  Fact 0.392: In 'Wizards vs. Everything', the wizards of the 13th floor come out, and Alex explains Dexter's evil.
  Fact 0.393: In 'Wizards vs. Everything', Dexter reveals himself as Gorog and says he is trying to rebuild his army of darkness.
  Fact 0.394: In 'Wizards vs. Everything', Felix uses a spell that makes all the wizards from the 13th floor go with them, leaving Alex, Justin and Harper behind.
  Fact 0.395: In 'Wizards vs. Everything', Justin realizes only mortals can go through the trashcan.
  Fact 0.396: In 'Wizards vs. Everything', Harper arrives in the wizard lair after Alex convinces her to go through the trashcan.
  Fact 0.397: In 'Wizards vs. Everything', Gorog takes the kidnapped wizards to his evil lair to make them evil.
  Fact 0.398: In 'Wizards vs. Everything', Gorog tricks Hugh Normous into becoming one of the evil wizards in his army.
  Fact 0.399: In 'Wizards vs. Everything', Professor Crumbs fakes becoming evil when Gorog plans to make him evil.
  Fact 0.400: In 'Wizards vs. Everything', Harper contacts Max and tells him about Gorog.
  Fact 0.401: In 'Wizards vs. Everything', Max suggests they look for spells in the wizard books.
  Fact 0.402: In 'Wizards vs. Everything', Gorog makes the wizard army dig a hole to the atmosphere of the wizard world, so they can take over it.
  Fact 0.403: In 'Wizards vs. Everything', Gorog tricks Mason into joining his army after Mason refuses.
  Fact 0.404: In 'Wizards vs. Everything', Gorog makes Felix send a black hole to suck Alex and Justin in.
  Fact 0.405: In 'Wizards vs. Everything', Gorog gives the kidnapped wizards their wands, but Crumbs refuses to let them take over the wizard world.
  Fact 0.406: In 'Wizards vs. Everything', Felix gets rid of Crumbs on Gorog's order.
  Fact 0.407: In 'Wizards vs. Everything', Crumbs arrives at the wizard lair as the black hole nearly sucks Alex and Justin in.
  Fact 0.408: In 'Wizards vs. Everything', Max suggests they put a black hole in the wizard lair.
  Fact 0.409: In 'Wizards vs. Everything', Max goes in and rescues Alex and Justin.
  Fact 0.410: In 'Wizards vs. Everything', Crumbs tells them to use their magic to stop Gorog.
  Fact 0.411: In 'Wizards vs. Everything', Mason tries to convince Alex to join Gorog.
  Fact 0.412: In 'Wizards vs. Everything', Gorog reveals Juliet, young again, Justin's ex-girlfriend, and she convinces Justin to join.
  Fact 0.413: In 'Wizards vs. Everything', Alex surrenders, and Max is forced to as well.
  Fact 0.414: In 'Wizards vs. Everything', Alex, Max and Justin use their combined magic to get rid of Gorog.
  Fact 0.415: In 'Wizards vs. Everything', all the supernatural beings are freed from Gorog's influence.
  Fact 0.416: In 'Wizards vs. Everything', Juliet and Justin get back together.
  Fact 0.417: In 'Wizards vs. Everything', Alex tricks Harper into moving into the Russos' basement by making her think all the homeless wizards and supernatural creatures will stay.
  Fact 0.418: Gregg Sulkin plays Mason as a special guest star in 'Wizards vs. Everything'.
  Fact 0.419: Bridgit Mendler guest stars as Juliet van Heusen, as special guest star, in 'Wizards vs. Everything'.
  Fact 0.420: Ian Abercrombie guest stars as Professor Crumbs in 'Wizards vs. Everything'.
  Fact 0.421: John Rubinstein guest stars as Gorog in 'Wizards vs. Everything'.
  Fact 0.422: Frank Pacheco guest stars as Felix in 'Wizards vs. Everything'.
  Fact 0.423: Fred Stoller guest stars as Dexter in 'Wizards vs. Everything'.
  Fact 0.424: Josh Sussman guest stars as Hugh Normous in 'Wizards vs. Everything'.
  Fact 0.425: Christopher Douglas Reed guest stars as Ogre in 'Wizards vs. Everything'.
  Fact 0.426: 'Wizards vs. Everything' is part 4 of a 4-part arc.
  Fact 0.427: Maria Canals Barrera, who plays Theresa Russo, was absent from 'Wizards vs. Everything'.
  Fact 0.428: David DeLuise, who plays Jerry Russo, was absent from 'Wizards vs. Everything'.
  Fact 0.429: Episode 104, 'Rock Around the Clock', was directed by Guy Distad and written by Robert Boesel, and aired November 4, 2011 with production code 414 and 3.8 million U.S. viewers.
  Fact 0.430: In 'Rock Around the Clock', the landlord of the Russos' building decides to sell the sub shop, evicting the Russos from the shop, their apartment and their lair, which holds their portal.
  Fact 0.431: In 'Rock Around the Clock', without the portal the Russos lose all contact with the wizard world.
  Fact 0.432: In 'Rock Around the Clock', the Russo family and Harper travel back to 1957 to stop Jerry's father from selling the shop to the landlord.
  Fact 0.433: In 'Rock Around the Clock', Jerry's father agrees not to sell the shop, and the Russos return to present day.
  Fact 0.434: In 'Rock Around the Clock', the Russos accidentally leave Harper behind in 1957 when she is distracted by a woman with a poodle.
  Fact 0.435: In 'Rock Around the Clock', Harper suggests the woman put a picture of a poodle on her skirt, creating the poodle skirt.
  Fact 0.436: In 'Rock Around the Clock', back in present day the sub shop is obsolete, boarded up and empty, with the subway car and the lair missing.
  Fact 0.437: In 'Rock Around the Clock', the Russos realize they messed with the fabric of time.
  Fact 0.438: In 'Rock Around the Clock', Theresa realizes they left Harper in 1957 and says they will have to go back.
  Fact 0.439: In 1957, Jerry's father reveals he did not sell the restaurant, but the restaurant failed to make enough business and he was forced to close it.
  Fact 0.440: In 'Rock Around the Clock', Justin realizes that when Jerry's father lost the restaurant, he moved out of the building, so the lair disappeared in present day.
  Fact 0.441: In 'Rock Around the Clock', Harper enrolled at Tribeca Prep as soon as she was forgotten in 1957, because she did not want to ruin her perfect attendance record.
  Fact 0.442: In 'Rock Around the Clock', Harper decides to stay in the past because she finds new popularity in 1957.
  Fact 0.443: In 'Rock Around the Clock', Max introduces the high-five to several students at Tribeca.
  Fact 0.444: In 'Rock Around the Clock', Alex realizes that making the sub shop the 1957 high school hangout will improve business.
  Fact 0.445: In 'Rock Around the Clock', the jukebox is broken, and Max plugs his MP3 player into the jukebox.
  Fact 0.446: In 'Rock Around the Clock', Harper uses her popularity to get the students dancing and having fun at the shop, and business peaks.
  Fact 0.447: In 'Rock Around the Clock', the Russos return to present day, where the lair and the sub shop are back.
  Fact 0.448: In 'Rock Around the Clock', Jerry owns the entire building, and the landlord now owns a Janitor service instead.
  Fact 0.449: In 'Rock Around the Clock', the high-five is now called a Max.
  Fact 0.450: In 'Rock Around the Clock', Alex remembers they forgot Harper in 1957, and they go back and get her.
  Fact 0.451: In 'Rock Around the Clock', the Russos accidentally end up in 1977 twice, thanks to Justin.
  Fact 0.452: Brendan Bradley guest stars as Hank in 'Rock Around the Clock'.
  Fact 0.453: Carlo Michael Mancini guest stars as Lenny Hune in 'Rock Around the Clock'.
  Fact 0.454: Episode 105, 'Harperella', was directed by Victor Gonzalez and written by Richard Goodman and Justin Varava, and aired November 18, 2011 with production code 427 and 3.5 million U.S. viewers.
  Fact 0.455: In 'Harperella', Harper is reading a storybook to children on Waverly Place when Alex pushes her away.
  Fact 0.456: In 'Harperella', the Wizard Storybook has the power of turning what the person reads real.
  Fact 0.457: In 'Harperella', Harper is teleported into the loft, where Theresa, her wicked stepmother, calls her Harperella and makes her do chores.
  Fact 0.458: In 'Harperella', Justin and Max are Harper's evil stepbrothers.
  Fact 0.459: In 'Harperella', Alex appears as Harper's fairy godmother.
  Fact 0.460: In 'Harperella', Max says Prince Charming has decided to use the Substation as the ballroom for the royal party.
  Fact 0.461: In 'Harperella', Alex uses a spell to turn Harper's bookdress into a princess gown and glass bowls into glass slippers.
  Fact 0.462: In 'Harperella', Alex turns the hot dog stand into a carriage to take Harper to the party.
  Fact 0.463: In 'Harperella', Harper dances with Prince Charming, Zeke.
  Fact 0.464: In 'Harperella', at the stroke of midnight Harper leaves her glass slipper behind.
  Fact 0.465: In 'Harperella', Justin, Max and Zeke, with pig noses, rush into the substation being chased by the big bad wolf, Mason.
  Fact 0.466: In 'Harperella', Cinderella and the three little pigs stories were blended together because some pages were missing.
  Fact 0.467: In 'Harperella', Alex tries a page from Max's favorite magic trick book, and David Copperfield appears and does magic tricks.
  Fact 0.468: In 'Harperella', Alex tries a page from Justin's science book, and a caveman appears who has a thing for Harper.
  Fact 0.469: In 'Harperella', the caveman is jealous and tries to attack David Copperfield.
  Fact 0.470: In 'Harperella', Alex finds the right pages and the story goes back to Cinderella.
  Fact 0.471: In 'Harperella', Theresa tries to wear the slipper but fails, and Harper fits it.
  Fact 0.472: In 'Harperella', Harper and Zeke dance at the royal party, and the world returns to normal.
  Fact 0.473: In 'Harperella', Alex planned for Harper and Zeke to dance, but they were really bad, so the ball dance turned into a clogging dance.
  Fact 0.474: David Copperfield guest stars as himself in 'Harperella'.
  Fact 0.475: Gregg Sulkin plays Mason as a special guest star in 'Harperella'.
  Fact 0.476: Dan Benson guest stars as Zeke in 'Harperella'.
  Fact 0.477: David DeLuise, who plays Jerry Russo, was absent from 'Harperella'.
  Fact 0.478: Episode 106, 'Who Will Be the Family Wizard', was directed by Victor Gonzalez and written by Vince Cheung and Ben Montanio for Part 1 and Todd J. Greenwald for Part 2, and aired January 6, 2012 with production codes 428 and 429 and 9.8 million U.S. viewers.
  Fact 0.479: In 'Who Will Be the Family Wizard', Professor Crumbs visits during a dinner Alex made for the family and reveals the Russo children can finally have their family wizard competition.
  Fact 0.480: In 'Who Will Be the Family Wizard', round one of the competition is a trivia round about magic.
  Fact 0.481: In 'Who Will Be the Family Wizard', Zeke smelled a purple substance in the lair and turned purple.
  Fact 0.482: In 'Who Will Be the Family Wizard', a griffin kidnaps Zeke and Harper.
  Fact 0.483: In 'Who Will Be the Family Wizard', Alex, Justin and Max use their three time-outs to rescue their friends.
  Fact 0.484: In 'Who Will Be the Family Wizard', the Russo children are disqualified for being gone too long.
  Fact 0.485: In 'Who Will Be the Family Wizard', the Russos lose their magic and become mortals when the lair disappears.
  Fact 0.486: In 'Who Will Be the Family Wizard', Justin and Max begin to hate Alex.
  Fact 0.487: In 'Who Will Be the Family Wizard', Jerry decides to sell the sub shop.
  Fact 0.488: In 'Who Will Be the Family Wizard', the kids go a day without magic and reopen and run the shop.
  Fact 0.489: In 'Who Will Be the Family Wizard', the host reveals the griffin attack was meant for Professor Crumbs and the weeks without magic were tests of the family bond.
  Fact 0.490: In 'Who Will Be the Family Wizard', the final round of the competition is a massive labyrinth where the first out wins.
  Fact 0.491: In 'Who Will Be the Family Wizard', Justin makes it out first and is declared the Family Wizard.
  Fact 0.492: In 'Who Will Be the Family Wizard', Justin declines the title because Alex was about to get out first and came back to help him.
  Fact 0.493: In 'Who Will Be the Family Wizard', Alex is declared the true family wizard.
  Fact 0.494: In 'Who Will Be the Family Wizard', Professor Crumbs retires as Headmaster of WizTech and declares Justin the new headmaster, granting him full wizard powers.
  Fact 0.495: In 'Who Will Be the Family Wizard', Jerry decides to pass down the sub shop to Max one day, and Max happily accepts.
  Fact 0.496: In 'Who Will Be the Family Wizard', the Russo family ends the episode hugging and gathering together, with Alex saying they are all finally happy at the same time.
  Fact 0.497: Gregg Sulkin plays Mason as a special guest star in 'Who Will Be the Family Wizard'.
  Fact 0.498: Bridgit Mendler guest stars as Juliet van Heusen, as special guest star, in 'Who Will Be the Family Wizard'.
  Fact 0.499: Dan Benson guest stars as Zeke in 'Who Will Be the Family Wizard'.
  Fact 0.500: Ian Abercrombie guest stars as Professor Crumbs in 'Who Will Be the Family Wizard'.
  Fact 0.501: Andy Kindler guest stars as Chancellor Tootietootie in 'Who Will Be the Family Wizard'.
  Fact 0.502: 'Who Will Be the Family Wizard' is an hour-long special.
Sentence 1: In the episode , `` Three Maxes and a Little Lady '' , Alex and Justin accidentally turn Max into a little girl , Maxine , portrayed by Bailee Madison . She ends up making Alex jealous when their father prefers Maxine over Alex . Maxine remains until `` Back to Max '' . In the episode `` Everything 's Rosie for Justin '' , Justin falls in love with Rosie , a guardian angel , but at the end of `` Dancing with Angels '' she becomes an Angel of Darkness . In `` Zeke Finds Out '' , the Russos finally reveal to Zeke that they are wizards . In `` Wizard of the Year '' , Alex is crowned Wizard of the Year for saving the world from the angels of darkness , and is reinstated in the Wizards Competition . In `` Justin 's Back In '' , Justin is reinstated in the Wizard Competition , after his class of delinquent wizards passes .
  Fact 1.0: In the episode 'Three Maxes and a Little Lady', Alex and Justin accidentally turn Max into a little girl named Maxine.
  Fact 1.1: Maxine is portrayed by Bailee Madison.
  Fact 1.2: Maxine makes Alex jealous when their father prefers Maxine over Alex.
  Fact 1.3: Maxine remains a little girl until 'Back to Max'.
  Fact 1.4: In 'Everything's Rosie for Justin', Justin falls in love with Rosie, a guardian angel.
  Fact 1.5: At the end of 'Dancing with Angels', Rosie becomes an Angel of Darkness.
  Fact 1.6: In 'Zeke Finds Out', the Russos finally reveal to Zeke that they are wizards.
  Fact 1.7: In 'Wizard of the Year', Alex is crowned Wizard of the Year for saving the world from the angels of darkness.
  Fact 1.8: In 'Wizard of the Year', Alex is reinstated in the Wizards Competition.
  Fact 1.9: In 'Justin's Back In', Justin is reinstated in the Wizard Competition after his class of delinquent wizards passes.

Retrieved items:
<item 1>
Wizards of Waverly Place (season 4)
In the end Mason 's parents like her as a werewolf and she says to him that they do not know the real her . She begins to leave and he says that he truly loves
Alex to his parents and that she is a wizard . She changes back to a wizard and Mason turns back to human form . After the parents find out they try to eat her
and Mason and Alex make a near escape . Meanwhile Max makes his parents ' food taste like kid food in order to make it taste good . Afterwards Jerry and Theresa
eat it , and as a result they act like little kids . Justin thinks he will have to fix it but Max comments on how he does not need help , as he will be the
future family wizard . Max uses another spell on cookies to make them older , but they turn into teenagers . Finally , Justin uses a spell on a pizza to turn
them back into their regular selves .
</item 1>

<item 2>
Wizards of Waverly Place (season 4)
Too fast for Maxine , Justin beats her , and Maxine is hurt ( physically ) . Jerry and Theresa rush to her side , caring for Maxine more than they care that
Justin beat Maxine in karate . Justin leaves , and Alex leaves after him , heartbroken . Back at home , Jerry tells Alex that he fell for Maxine because she
reminded him of Alex when she was little and they bond again . Meanwhile , Zeke thinks the Russos hate him , but really it was just Harper 's alibi in order to
keep him away from the Sub Station so as not to make him suspicious about Max 's disappearance and Maxine 's appearance . Guest stars : Bailee Madison as Maxine
, Dan Benson as Zeke , Lanny Joon as Sensei Absent : Jake T. Austin as Max Russo 86 7 `` Everything 's Rosie for Justin '' Victor Gonzalez Justin Varava
February 4 , 2011 ( 2011 - 02 - 04 ) 407 3.9 Justin and his delinquent class have a wizard wand dance assessment to do which will get them back in Wiztech .
</item 2>

<item 3>
Wizards of Waverly Place (season 4)
After her first day of school , Maxine laments that she has no friends , which leads Harper to insist Maxine have a slumber party ( which is really what Harper
wants because she has never had a slumber party ) , that eventually becomes a disaster . Tina reveals that Rosie was a good angel , and her teacher however ,
Gorog reached her before her transport arrived and corrupted her . Alex and Tina go to the Dark Realm to save Justin . Alex borrows a pair of wings from the
guardian angels . Alex takes the Moral Compass and tries to turn it to good but Justin stops her and then Rosie used her good angel side and they fly up and
when they land , Justin turns the compass to good . Tina and Alex take it and run . When Rosie and Justin kiss , Rosie turns into a good angel ( with white
wings ) , and Justin loses the dark wings . Guest stars : Bill Chott as Mr.
</item 3>

<item 4>
Wizards of Waverly Place (season 4)
Guest stars : Bill Chott as Mr. Laritate , Ian Abercrombie as Professor Crumbs , Bailee Madison as Maxine , Cameron Sanders as Nelvis , LJ Benet as Little
Crumbs , Jackie Evancho as Choir Girl Note : Jake T. Austin returns as Max near the end of the episode . Bailee Madison marks her final appearance as Maxine in
this episode . 90 11 `` Zeke Finds Out '' David DeLuise Peter Dirksen April 8 , 2011 ( 2011 - 04 - 08 ) 412 3.0 When Alex gets bored of Zeke 's failing magic
tricks , she decides to use magic to make Zeke 's tricks come true . Everything goes downhill when Zeke tells Harper he believes he is a wizard and is now
booking a magic act at the sub station for a kid 's birthday party . Since Justin and Harper are tired of Alex ruining things all the time , they decide to
leave everything up to her .
</item 4>

<item 5>
Wizards of Waverly Place (season 4)
Alex , Justin and Max use their three time - outs to go and rescue their friends . They make it back , only to discover that they have been gone too long and
have been disqualified from the competition . They all return home , where the lair disappears and they all lose their magic , thus becoming mortals . Justin
and Max begin to hate Alex because of her insistence on saving Zeke and Harper resulting in them losing . Convinced that the loss of their powers has ruined the
family , Jerry decides to sell the sub shop . The kids go for a day without magic , during which , slowly gaining back trust in each other as they reopen and
run the shop . Suddenly , they are returned to the area of the Wizard Competition , where the host reveals that the griffin attack ( which was actually meant
for Professor Crumbs ) and the weeks they went without magic were tests to test their family bond . They enter the final round of the competition , a massive
labyrinth . The first out of the maze will win . Justin makes it out first and is declared the Family Wizard .
</item 5>

<item 6>
Wizards of Waverly Place (season 4)
They then shrink it and go into Mason 's mouth . While there , Alex , Max , and Justin see Mason 's thoughts , which are all of Alex . They get Dean out , and
Alex , when Harper , Max , and Justin leave , Alex tells Dean that he was knocked out and the doctor 's solution was for them all to dress up as cows . Dean
then gives her his offer once again : he wants to get back with her . Alex tells him ' no ' , and then gets back together with Mason , with deciding to get back
to the Wizard Competition . Special guest star : Gregg Sulkin as Mason Guest stars : Daniel Samonas as Dean , Frank Pacheco as Felix , Shoshana Bush as Tyler 84
5 `` Three Maxes and a Little Lady '' Victor Gonzalez Richard Goodman January 7 , 2011 ( 2011 - 01 - 07 ) 405 4.0 Alex joins Justin 's wizard class to try to
move up in the competition . Max joins a sophisticated wizard society . They suggest he move the wizard competition up and he moves it to a few days away .
</item 6>

<item 7>
Wizards of Waverly Place (season 4)
Justin takes Maxine to her ( Max 's ) karate class and Maxine kicks Justin 's butt . After the class , Alex decides to make Jerry 's favorite sandwich in order
to win his love back , but Maxine comes along with Justin , announcing that she beat Justin in karate , and Jerry immediately falls for her cuteness again .
More jealous than ever , Alex comes back with Justin to Maxine 's karate class with their parents , having a slight idea of knowing how to change Maxine back to
Max . They cast spells on Maxine , Justin using the spell he used on Max that turned him into Maxine , and Alex using the reverse spell . Unfortunately , it
only makes Maxine cuter , which makes her furious . Instead of going against the person she was supposed to , she demands that she kicks either Alex or Justin
's butt . Justin goes against her , but seeing that he would lose , Alex casts a spell on Justin to make him a karate master . Too fast for Maxine , Justin
beats her , and Maxine is hurt ( physically ) .
</item 7>

<item 8>
Wizards of Waverly Place (season 4)
Guest stars : Bill Chott as Mr. Laritate , Bailee Madison as Maxine , Leven Rambin as Rosie , John Rubinstein as Gorog , China McClain as Tina , Amy Tolsky as
Florence Note : This is a one - hour special . Absent : Jake T. Austin as Max Russo 89 10 `` Back To Max '' Guy Distad Todd J. Greenwald March 11 , 2011 ( 2011
- 03 - 11 ) 411 3.7 Justin invites Professor Crumbs to check on his delinquent wizards class , to see how their wizard training is progressing . During the
conversation , Maxine interrupts , causing Alex and Justin to panic since they do not know how to explain `` Maxine '' , and they try to find a way to turn
Maxine back to Max . Meanwhile , Harper takes a role in the school 's `` Spirit of America '' play , only to be replaced by Maxine . Guest stars : Bill Chott as
Mr.
</item 8>

<item 9>
Wizards of Waverly Place (season 4)
They suggest he move the wizard competition up and he moves it to a few days away . Max quits the society because he finds out that they are taking advantage of
him . Alex and Justin transform into Max in an attempt to change the date of the family wizard competition back to its original date and succeed . In the end of
the episode they accidentally turn Max indefinitely into a little girl . Guest stars : Bailee Madison as Maxine , Frank Pacheco as Felix , Kari Wahlgren as
Helen , Shane Harper as Fidel Note : Bailee Madison begins her recurring role as Maxine , the female Max in this episode . Jake T. Austin was absent in four
other episodes and returns at the end of the tenth episode `` Back to Max '' . 85 6 `` Daddy 's Little Girl '' Robbie Countryman Gigi McCreery & Perry Rein
January 21 , 2011 ( 2011 - 01 - 21 ) 406 3.4 With Max still a little girl , their parents pay more attention to Max ( Maxine ) , and Alex becomes jealous .
</item 9>

<item 10>
Wizards of Waverly Place (season 4)
She ends up making Alex jealous when their father prefers Maxine over Alex . Maxine remains until `` Back to Max '' . In the episode `` Everything 's Rosie for
Justin '' , Justin falls in love with Rosie , a guardian angel , but at the end of `` Dancing with Angels '' she becomes an Angel of Darkness . In `` Zeke Finds
Out '' , the Russos finally reveal to Zeke that they are wizards . In `` Wizard of the Year '' , Alex is crowned Wizard of the Year for saving the world from
the angels of darkness , and is reinstated in the Wizards Competition . In `` Justin 's Back In '' , Justin is reinstated in the Wizard Competition , after his
class of delinquent wizards passes . Meanwhile , Alex 's relationship with Mason is jeopardized again when a beast tamer , Chase Riprock , falls in love with
Alex and Mason gets jealous . This results in Alex breaking up with Mason . Alex , Justin and Max then save the world from being destroyed by an asteroid .
Later , Alex and Harper move into an apartment building with a secret 13th floor for wizards and other creatures in the wizard world .
</item 10>
=== END ITEM id=c90252338c16c857 ===

=== ITEM id=a8f0276d5cfd4e2e ===
Gold sentences and facts:
Sentence 0: In negotiation theory , the best alternative to a negotiated agreement or BATNA is the most advantageous alternative course of action a party can take if negotiations fail and an agreement can not be reached . BATNA is the key focus and the driving force behind a successful negotiator . A party should generally not accept a worse resolution than its BATNA . Care should be taken , however , to ensure that deals are accurately valued , taking into account all considerations , such as relationship value , time value of money and the likelihood that the other party will live up to their side of the bargain . These other considerations are often difficult to value , since they are frequently based on uncertain or qualitative considerations , rather than easily measurable and quantifiable factors .
  Fact 0.0: In negotiation theory, the best alternative to a negotiated agreement (BATNA) is the most advantageous course of action a party can take if negotiations fail and no agreement can be reached.
  Fact 0.1: BATNA is the key focus of a successful negotiator.
  Fact 0.2: BATNA is the driving force behind a successful negotiator.
  Fact 0.3: A party should generally not accept a resolution worse than its BATNA.
  Fact 0.4: Care should be taken to ensure that deals are accurately valued.
  Fact 0.5: Deal valuation should take into account all considerations.
  Fact 0.6: Relationship value is a consideration in valuing a deal.
  Fact 0.7: Time value of money is a consideration in valuing a deal.
  Fact 0.8: The likelihood that the other party will live up to its side of the bargain is a consideration in valuing a deal.
  Fact 0.9: The other considerations are often difficult to value.
  Fact 0.10: The other considerations are frequently based on uncertain or qualitative considerations.
  Fact 0.11: The other considerations are qualitative rather than easily measurable and quantifiable factors.

Retrieved items:
<item 1>
Best alternative to a negotiated agreement
History ( edit ) BATNA was developed by negotiation researchers Roger Fisher and William Ury of the Harvard Program on Negotiation ( PON ) , in their series of
books on Principled negotiation that started with Getting to YES , unwittingly duplicating the game theory concept of a disagreement point from bargaining
problems pioneered by Nobel Laureate John Forbes Nash decades earlier . A Nash Equilibrium is reached among a group of players when no player can benefit from
changing strategies if every other player sticks to their current strategy . For example , Amy and Phil are in Nash Equilibrium if Amy is making the best
decision she can , taking into account Phil 's decision , and Phil is making the best decision he can , taking into account Amy 's decision . Likewise , a group
of players are in Nash Equilibrium if each one is making the best decision that he or she can , taking into account the decisions of the others .
</item 1>

<item 2>
Best alternative to a negotiated agreement
See also ( edit ) Conflict resolution research Opportunity cost References ( edit ) Jump up ^ Best Alternative to a Negotiated Agreement Negotiation Experts
Jump up ^ Best Alternative to a Negotiated Agreement YArooms Jump up ^ Negotiation , Readings , Exercises and Cases , Roy J. Lewicki ^ Jump up to : Nolan -
Haley , Jaqueline M ( 2001 ) . Alternative Dispute Resolution in a Nutshell . Thomson West . pp. 39 -- 50 . ISBN 978 - 0 - 314 - 18014 - 8 . ^ Jump up to :
Fisher and Ury , Roger and William ( 2011 ) . Getting to YES . Penguin Books . pp. 1 -- 170 . ISBN 978 - 0 - 14 - 311875 - 6 . Jump up ^ Gulliver , P.H ( 1979 )
. Disputes and Negotiation : A Cross Culture Perspective . Academic Press . p. 287 . Jump up ^ Morris , Michael W. ; Gelfand , Michele J. ( 2004 ) .
</item 2>

<item 3>
Best alternative to a negotiated agreement
, a non-profit organization . About Wikipedia
</item 3>

<item 4>
Best alternative to a negotiated agreement
Retrieved 1 October 2012 . External links ( edit ) BATNA Explained by David Venter Alternative definition of BATNA Retrieved from ``
https://en.wikipedia.org/w/index.php?title=Best_alternative_to_a_negotiated_agreement&oldid=819966880 '' Categories : Negotiation Dispute resolution Business
terms Hidden categories : Articles needing more detailed references Articles needing additional references from July 2013 All articles needing additional
references All articles lacking reliable references Articles lacking reliable references from July 2013 All articles with unsourced statements Articles with
unsourced statements from October 2007 Articles lacking reliable references from February 2015 Wikipedia articles needing page number citations from July 2013
Talk Contents About Wikipedia Deutsch Español Հայերեն Polski Edit links This page was last edited on 12 January 2018 , at 08 : 19 . Text is available under the
Creative Commons Attribution - ShareAlike License ; additional terms may apply . By using this site , you agree to the Terms of Use and Privacy Policy .
Wikipedia ® is a registered trademark of the Wikimedia Foundation , Inc. , a non-profit organization .
</item 4>

<item 5>
Best alternative to a negotiated agreement
Others may employ a soft style , which is friendly , trusting , compromising , and conflict avoiding . According to Fisher and Ury , when hard negotiators meet
soft negotiators , the hard negotiators usually win their position , but at the cost of potentially damaging the long term relationship between the parties .
Attractive alternatives are needed to develop a strong BATNA . In the best - selling book Getting to YES : Negotiating Agreement Without Giving In , the authors
give 3 suggestions of how to accomplish this : Inventing a list of actions one might take if no agreement is reached Converting some of the more promising ideas
and transforming them into tangible and partial alternatives Selecting the alternative that sounds best In negotiations involving different cultures , all
parties need to account for cultural cognitive behaviors and should not let judgments and biases affect the negotiation . The individual should be separate from
the objective . The purpose here , as Gulliver mentions , is for negotiation parties to be aware . Preparation at all levels , including prejudice - free
thoughts , emotion - free behavior , bias - free behavior are helpful according to Morris and Gelfand .
</item 5>

<item 6>
Best alternative to a negotiated agreement
Contents ( hide ) 1 Overview 2 History 3 See also 4 References 5 External links Overview ( edit ) The BATNA is often seen by negotiators not as a safety net ,
but rather as a point of leverage in negotiations . Although a negotiator 's alternative options should , in theory , be straightforward to evaluate , the
effort to understand which alternative represents a party 's BATNA is often not invested . Options need to be actual and actionable to be of value , however
without the investment of time , options will frequently be included that fail on one of these criteria . Most managers overestimate their BATNA whilst
simultaneously investing too little time into researching their real options . This can result in poor or faulty decision making and negotiating result .
Negotiators also need to be aware of the other negotiator 's BATNA and to identify how it compares to what they are offering . Some people may adopt aggressive
, coercive , threatening and / or deceptive techniques , this is known as a hard negotiation style ; a theoretical example of this is adversarial approach style
negotiation . Others may employ a soft style , which is friendly , trusting , compromising , and conflict avoiding .
</item 6>

<item 7>
Reinforcement
In 1970 , Baer and Wolf created a name for the use of natural reinforcers called `` behavior traps '' . A behavior trap requires only a simple response to enter
the trap , yet once entered , the trap can not be resisted in creating general behavior change . It is the use of a behavioral trap that increases a person 's
repertoire , by exposing them to the naturally occurring reinforcement of that behavior . Behavior traps have four characteristics : They are `` baited '' with
virtually irresistible reinforcers that `` lure '' the student to the trap Only a low - effort response already in the repertoire is necessary to enter the trap
Interrelated contingencies of reinforcement inside the trap motivate the person to acquire , extend , and maintain targeted academic / social skills They can
remain effective for long periods of time because the person shows few , if any , satiation effects As can be seen from the above , artificial reinforcement is
in fact created to build or develop skills , and to generalize , it is important that either a behavior trap is introduced to `` capture '' the skill and
utilize naturally occurring reinforcement to maintain or increase it .
</item 7>

<item 8>
Bond (finance)
An alternative process for bond issuance , which is commonly used for smaller issues and avoids this cost , is the private placement bond . Bonds sold directly
to buyers may not be tradeable in the bond market . Historically an alternative practice of issuance was for the borrowing government authority to issue bonds
over a period of time , usually at a fixed price , with volumes sold on a particular day dependent on market conditions . This was called a tap issue or bond
tap . Features ( edit ) Principal ( edit ) 1978 $1,000 U.S. Treasury Bond Nominal , principal , par , or face amount is the amount on which the issuer pays
interest , and which , most commonly , has to be repaid at the end of the term . Some structured bonds can have a redemption amount which is different from the
face amount and can be linked to performance of particular assets . Maturity ( edit ) The issuer has to repay the nominal amount on the maturity date . As long
as all due payments have been made , the issuer has no further obligations to the bond holders after the maturity date .
</item 8>

<item 9>
Best alternative to a negotiated agreement
( 2004 ) . `` Cultural Differences and Cognitive Dynamics : Expanding the Cognitive Perspective on Negotiation '' . In Gelfand , Michele J. ; Brett , Jeanne M.
The Handbook of Negotiation and Culture . Stanford University Press . pp. 45 -- 70 . ISBN 978 - 0 - 8047 - 4586 - 4 . Jump up ^ Myerson , Roger B. `` Nash
Equilibrium and the History of Economic Theory '' ( PDF ) . Nash Equilibrium and the History of Economic Theory . Journal of Economic Literature . Retrieved 1
October 2012 . Jump up ^ Hawkins and Steiner , Jeff and Neil . `` The Nash Equilibrium Meets Batna '' ( PDF ) . Gamed Therory Varied Uses in ADR . Harvard
University Press . Archived from the original ( PDF ) on February 28 , 2013 . Retrieved 1 October 2012 .
</item 9>

<item 10>
Best alternative to a negotiated agreement
Best alternative to a negotiated agreement - wikipedia Best alternative to a negotiated agreement Jump to : navigation , search This article needs additional
citations for verification . Please help improve this article by adding citations to reliable sources . Unsourced material may be challenged and removed . (
July 2013 ) ( Learn how and when to remove this template message ) In negotiation theory , the best alternative to a negotiated agreement or BATNA is the most
advantageous alternative course of action a party can take if negotiations fail and an agreement can not be reached . BATNA is the key focus and the driving
force behind a successful negotiator . A party should generally not accept a worse resolution than its BATNA . Care should be taken , however , to ensure that
deals are accurately valued , taking into account all considerations , such as relationship value , time value of money and the likelihood that the other party
will live up to their side of the bargain . These other considerations are often difficult to value , since they are frequently based on uncertain or
qualitative considerations , rather than easily measurable and quantifiable factors .
</item 10>
=== END ITEM id=a8f0276d5cfd4e2e ===

=== ITEM id=6891e23c3f79392f ===
Gold sentences and facts:
Sentence 0: The Nightmare Before Christmas originated in a poem written by Tim Burton in 1982 , while he was working as an animator at Walt Disney Feature Animation . With the success of Vincent in the same year , Walt Disney Studios began to consider developing The Nightmare Before Christmas as either a short film or 30 - minute television special . Over the years , Burton 's thoughts regularly returned to the project , and in 1990 , he made a development deal with Disney . Production started in July 1991 in San Francisco ; Disney released the film through its Touchstone Pictures banner because the studio believed the film would be `` too dark and scary for kids '' .
  Fact 0.0: The Nightmare Before Christmas originated in a poem written by Tim Burton in 1982.
  Fact 0.1: Tim Burton wrote the poem in 1982 while he was working as an animator at Walt Disney Feature Animation.
  Fact 0.2: Walt Disney Studios began to consider developing The Nightmare Before Christmas in the same year as the success of Vincent.
  Fact 0.3: Walt Disney Studios considered developing The Nightmare Before Christmas as either a short film or a 30-minute television special.
  Fact 0.4: Tim Burton's thoughts regularly returned to The Nightmare Before Christmas over the years.
  Fact 0.5: Tim Burton made a development deal with Disney in 1990.
  Fact 0.6: Production of The Nightmare Before Christmas started in July 1991 in San Francisco.
  Fact 0.7: Disney released The Nightmare Before Christmas through its Touchstone Pictures banner.
  Fact 0.8: Disney released the film through Touchstone because the studio believed the film would be 'too dark and scary for kids'.

Retrieved items:
<item 1>
The Nightmare Before Christmas
The film was met with both critical and financial success , grossing over $76 million during its initial run . It was nominated for the Academy Award for Best
Visual Effects , a first for an animated film . The film has since been reissued by Walt Disney Pictures , and was re-released annually in Disney Digital 3 - D
from 2006 until 2009 , making it the first stop - motion animated feature to be entirely converted to 3D . Contents ( hide ) 1 Plot 2 Voice cast 3 Production
3.1 Development 3.2 Filming 4 Soundtracks 5 Release 5.1 Home media 5.2 Marketing 6 Reception 6.1 Box office 6.2 Critical response 6.3 Accolades 7 Possible
sequel 8 Related media 9 See also 10 Notes 11 References 12 Further reading 13 External links Plot ( edit ) Halloween Town is a fantasy world filled with
citizens such as deformed monsters , ghosts , ghouls , goblins , zombies , mummies , vampires , werewolves , and witches .
</item 1>

<item 2>
The Nightmare Before Christmas
Nightmare marked Burton 's third film in a row to have a Christmas setting . Burton could not direct because of his commitment to Batman Returns and he did not
want to be involved with `` the painstakingly slow process of stop motion '' . To adapt his poem into a screenplay , Burton approached Michael McDowell , his
collaborator on Beetlejuice . McDowell and Burton experienced creative differences , which convinced Burton to make the film as a musical with lyrics and
compositions by frequent collaborator Danny Elfman . Elfman and Burton created a rough storyline and two - thirds of the film 's songs . Elfman found writing
Nightmare 's eleven songs as `` one of the easiest jobs I 've ever had . I had a lot in common with Jack Skellington . '' Caroline Thompson still had yet to be
hired to write the screenplay . With Thompson 's screenplay , Selick stated , `` there are very few lines of dialogue that are Caroline 's . She became busy on
other films and we were constantly rewriting , re-configuring and developing the film visually . ''
</item 2>

<item 3>
The Nightmare Before Christmas
of the King ( 2003 ) Spider - Man 2 ( 2004 ) Batman Begins ( 2005 ) Superman Returns ( 2006 ) Enchanted ( 2007 ) The Curious Case of Benjamin Button ( 2008 )
Watchmen ( 2009 ) Alice in Wonderland ( 2010 ) Harry Potter and the Deathly Hallows -- Part 2 ( 2011 ) Life of Pi ( 2012 ) Her ( 2013 ) The Hobbit : The Battle
of the Five Armies ( 2014 ) Cinderella ( 2015 ) The Jungle Book ( 2016 ) Disney Christmas films Theatrical shorts Empty Socks ( 1927 ) Mickey 's Orphans ( 1931
) Santa 's Workshop ( 1932 ) Mickey 's Good Deed ( 1932 ) The Night Before Christmas ( 1933 ) Broken Toys ( 1935 ) Toy Tinkers ( 1949 ) Pluto 's Christmas Tree
( 1952 ) The Small One ( 1978 ) Mickey 's Christmas Carol ( 1983 ) The Little Matchgirl ( 2006 ) Olaf 's Frozen Adventure ( 2017 ) Theatrical
</item 3>

<item 4>
The Nightmare Before Christmas
The Nightmare Before Christmas - wikipedia The Nightmare Before Christmas Jump to : navigation , search This article is about the film . For other uses , see
The Nightmare Before Christmas ( disambiguation ) . The Nightmare Before Christmas Theatrical release poster Directed by Henry Selick Produced by Tim Burton
Denise Di Novi Screenplay by Caroline Thompson Story by Michael McDowell ( adaptation ) Based on Story and characters by Tim Burton Starring Chris Sarandon
Danny Elfman Catherine O'Hara William Hickey Glenn Shadix Paul Reubens Ken Page Ed Ivory Music by Danny Elfman Cinematography Pete Kozachik Edited by Stan Webb
Production companies Touchstone Pictures Skellington Productions Distributed by Buena Vista Pictures Release date October 29 , 1993 ( 1993 - 10 - 29 ) Running
time 76 minutes Country United States Language English Budget $18 million Box office $76.2 million The Nightmare Before Christmas ( also known as Tim Burton 's
The Nightmare Before Christmas ) is a 1993 American stop - motion animated musical dark fantasy film directed by Henry Selick , and produced and conceived by
Tim Burton .
</item 4>

<item 5>
The Nightmare Before Christmas
It tells the story of Jack Skellington , a resident from `` Halloween Town '' who stumbles through a portal to `` Christmas Town '' and decides to celebrate the
holiday , with some dastardly and comical consequences . Danny Elfman wrote the songs and score , and provided the singing voice of Jack . The principal voice
cast also includes Chris Sarandon , Catherine O'Hara , William Hickey , Ken Page , Paul Reubens , and Glenn Shadix . The Nightmare Before Christmas originated
in a poem written by Tim Burton in 1982 , while he was working as an animator at Walt Disney Feature Animation . With the success of Vincent in the same year ,
Walt Disney Studios began to consider developing The Nightmare Before Christmas as either a short film or 30 - minute television special . Over the years ,
Burton 's thoughts regularly returned to the project , and in 1990 , he made a development deal with Disney . Production started in July 1991 in San Francisco ;
Disney released the film through its Touchstone Pictures banner because the studio believed the film would be `` too dark and scary for kids '' .
</item 5>

<item 6>
The Nightmare Before Christmas
`` I think those are some of the most inventive moments in cartoon history , in no way racist , even though he was sometimes a villain . We went with Ken Page ,
who is a black singer , and he had no problem with it '' . Accolades ( edit ) The film was nominated for both the Academy Award for Best Visual Effects and the
Hugo Award for Best Dramatic Presentation . Nightmare won the Saturn Award for Best Fantasy Film , while Elfman won Best Music . Selick and the animators were
also nominated for their work . Elfman was nominated for the Golden Globe Award for Best Original Score . Most recently , the film ranked # 1 on Rotten Tomatoes
' `` Top 25 Best Christmas Movies '' list . The film is recognized by American Film Institute in these lists : 2008 : AFI 's 10 Top 10 : Nominated Animation
Film Possible sequel ( edit ) In 2001 , Disney began to consider producing a sequel , but rather than using stop motion , Disney wanted to use computer
animation . Burton convinced Disney to drop the idea .
</item 6>

<item 7>
The Nightmare Before Christmas
Other songs included are original holiday songs changed to incorporate the theme of the movie . The last song on the list , however , is the soundtrack for the
Disneyland Haunted Mansion Holiday ride . Release ( edit ) Disney decided to release the film under their Touchstone Pictures banner because they thought the
film would be `` too dark and scary for kids '' , Selick remembered . `` Their biggest fear , and why it was kind of a stepchild project , ( was ) they were
afraid of their core audience hating the film and not coming . '' To convey Burton 's involvement and attract a wider audience , Disney marketed the film as Tim
Burton 's The Nightmare Before Christmas . Burton explained that , `` ... it turned more into more of a brand - name thing , it turned into something else ,
which I 'm not quite sure about . '' The film premiered at the New York Film Festival on October 9 , and was given a limited release on October 15 , 1993 ,
before its wide theatrical release on October 29 .
</item 7>

<item 8>
The Nightmare Before Christmas
This dazzling mix of fun and fright also explodes the notion that animation is kid stuff. ... It 's 74 minutes of timeless movie magic . '' James Berardinelli
stated `` The Nightmare Before Christmas has something to offer just about everyone . For the kids , it 's a fantasy celebrating two holidays . For the adults ,
it 's an opportunity to experience some light entertainment while marveling at how adept Hollywood has become at these techniques . There are songs , laughs ,
and a little romance . In short , The Nightmare Before Christmas does what it intends to : entertain . '' Desson Thomson of The Washington Post enjoyed
stylistic features in common with Oscar Wilde , German Expressionism , the Brothers Grimm and The Cabinet of Dr. Caligari . Michael A. Morrison discusses the
influence of Dr. Seuss ' How the Grinch Stole Christmas ! on the film , writing that Jack parallels the Grinch and Zero parallels Max , the Grinch 's dog .
Philip Nel writes that the film `` challenges the wisdom of adults through its trickster characters '' , contrasting Jack as a `` good trickster '' with Oogie
Boogie , whom he also compares with Dr.
</item 8>

<item 9>
The Nightmare Before Christmas
He created concept art and storyboards for the project in collaboration with Rick Heinrichs , who also sculpted character models ; Burton later showed his and
Heinrichs ' works - in - progress to Henry Selick , also a Disney animator at the time . After the success of Vincent in 1982 , Disney started to consider
developing The Nightmare Before Christmas as either a short film or 30 - minute holiday television special . However , the project 's development eventually
stalled , as its tone seemed `` too weird '' to the company . As Disney was unable to `` offer his nocturnal loners enough scope '' , Burton was fired from the
studio in 1984 , and went on to direct the commercially successful films Beetlejuice and Batman . Director Henry Selick ( left ) and producer Tim Burton ( right
) on the Nightmare Before Christmas set . Over the years , Burton regularly thought about the project . In 1990 , Burton found out that Disney still owned the
film rights . He and Selick committed to produce a full - length film with the latter as director . Disney was looking forward to Nightmare `` to show
capabilities of technical and storytelling achievements that were present in Who Framed Roger Rabbit . ''
</item 9>

<item 10>
The Nightmare Before Christmas
E. Disney Walt Disney Don Hahn Jeffrey Katzenberg John Lasseter Peter Schneider Thomas Schumacher David Stainton Disney 's Nine Old Men Les Clark Marc Davis
Ollie Johnston Milt Kahl Ward Kimball Eric Larson John Lounsbery Wolfgang Reitherman Frank Thomas Related topics History Disney animators ' strike Disney
Renaissance Methods and technologies 12 basic principles of animation Computer Animation Production System Disney Animation : The Illusion of Life Multiplane
camera Documentaries Frank and Ollie ( 1995 ) The Sweatbox ( 2001 ) Dream On Silly Dreamer ( 2005 ) Waking Sleeping Beauty ( 2009 ) Other Disney animation units
Disney Television Animation DisneyToon Studios ( WDAS unit ) Lucasfilm Animation Marvel Animation Pixar Animation Studios Circle 7 ( defunct ) Miscellaneous
Alice Comedies Laugh - O - Gram Studio List of Disney animated shorts List of Disney theatrical animated features unproduced Oswald the Lucky Rabbit Mickey
Mouse ( film series ) Silly Symphonies Once Upon a Time Saturn Award for Best Fantasy Film The Golden Voyage of Sinbad ( 1973 ) Doc Savage : The Man of Bronze (
1974 / 75 ) The Holes ( 1976 ) Oh ,
</item 10>
=== END ITEM id=6891e23c3f79392f ===
