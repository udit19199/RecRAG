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

=== ITEM id=6fd49e8b71193d84 ===
Gold sentences and facts:
Sentence 0: Central London Property Trust Ltd v High Trees House Ltd ( 1947 ) KB 130 ( or the High Trees case ) is an English contract law decision in the High Court . It reaffirmed and extended the doctrine of promissory estoppel in contract law in England and Wales . However , the most significant part of the judgment is obiter dicta as it relates to hypothetical facts ; that is , the landlord did not seek repayment of the full wartime rent .
  Fact 0.0: Central London Property Trust Ltd v High Trees House Ltd is cited as (1947) KB 130.
  Fact 0.1: Central London Property Trust Ltd v High Trees House Ltd is also called the High Trees case.
  Fact 0.2: Central London Property Trust Ltd v High Trees House Ltd is an English contract law decision in the High Court.
  Fact 0.3: The decision reaffirmed the doctrine of promissory estoppel in contract law in England and Wales.
  Fact 0.4: The decision extended the doctrine of promissory estoppel in contract law in England and Wales.
  Fact 0.5: The most significant part of the judgment is obiter dicta.
  Fact 0.6: The obiter dicta relates to hypothetical facts.
  Fact 0.7: In the hypothetical facts, the landlord did not seek repayment of the full wartime rent.

Retrieved items:
<item 1>
Central London Property Trust Ltd v High Trees House Ltd
If I were to consider this matter without regard to recent developments in the law , there is no doubt that had the plaintiffs claimed it , they would have been
entitled to recover ground rent at the rate of £ 2,500 a year from the beginning of the term , since the lease under which it was payable was a lease under seal
which , according to the old common law , could not be varied by an agreement by parol ( whether in writing or not ) , but only by deed . Equity , however
stepped in , and said that if there has been a variation of a deed by a simple contract ( which in the case of a lease required to be in writing would have to
be evidenced by writing ) , the courts may give effect to it as is shown in Berry v. Berry ( 1929 ) 2 K.B. 316 . That equitable doctrine , however , could
hardly apply in the present case because the variation here might be said to have been made without consideration . With regard to estoppel , the representation
made in relation to reducing the rent was not a representation of an existing fact .
</item 1>

<item 2>
Central London Property Trust Ltd v High Trees House Ltd
In Amalgamated Investment Co v Texas Bank it was decided that proprietory estoppel can act as a sword and not merely as a shield ( that is , it can be used as a
cause of action rather than merely providing a defence to an action ) . See also ( edit ) English contract law Estoppel Promise References ( edit ) Jump up ^
Collier v P & MJ Wright ( Holdings ) Ltd ( 2007 ) EWCA Civ 1329 Jump up ^ Amalgamated Investment Co v Texas Bank , ( 1982 ) EWHC 84 ( QB ) .
</item 2>

<item 3>
Central London Property Trust Ltd v High Trees House Ltd
I am satisfied that the promise was understood by all parties only to apply under the conditions prevailing at the time when it was made , namely , when the
flats were only partially let , and that it did not extend any further than that . When the flats became fully let , early in 1945 , the reduction ceased to
apply . In those circumstances , under the law as I hold it , it seems to me that rent is payable at the full rate for the quarters ending September 29 and
December 25 , 1945 . If the case had been one of estoppel , it might be said that in any event the estoppel would cease when the conditions to which the
representation applied came to an end , or it also might be said that it would only come to an end on notice . In either case it is only a way of ascertaining
what is the scope of the representation . I prefer to apply the principle that a promise intended to be binding , intended to be acted on and in fact acted on ,
is binding so far as its terms properly apply .
</item 3>

<item 4>
Central London Property Trust Ltd v High Trees House Ltd
Here it was binding as covering the period down to the early part of 1945 , and as from that time full rent is payable . I therefore give judgment for the
plaintiff company for the amount claimed . Significance ( edit ) Advances have been made in promissory estoppel since its inception in High Trees to create a
new inroad into the rule in Pinnel 's case that an agreement to accept part payment of a debt in full satisfaction of it is unenforceable for want of
consideration . Denning commented that such an agreement should now be enforceable under the doctrine of promissory estoppel , and indeed the plaintiff did not
seek the full debt on the basis of what was fair and , perhaps , thought was the law . However , the courts were at first reluctant to overrule or distinguish
cases like Pinnel 's case and Foakes v Beer having formed part of the common law for so long . Lady Justice Arden in Collier v P & MJ Wright ( Holdings ) Ltd (
2007 ) accepted in principle that High Trees could be used to extinguish a creditor 's right to full payment of a debt in such circumstances .
</item 4>

<item 5>
Central London Property Trust Ltd v High Trees House Ltd
for last two quarters of 1945 ) . Judgment ( edit ) Based on previous judgments as Hughes v Metropolitan Railway Co , Denning J held that the full rent was
payable from the time that the flats became fully occupied in mid-1945 . However , he continued in an obiter statement that if Central London had tried to claim
for the full rent from 1940 onwards , they would not have been able to . This was reasoned on the basis that if a party leads another party to believe that he
will not enforce his strict legal rights , then the Courts will prevent him from doing so at a later stage . Being obiter dicta and in a court of first instance
this was doubly not a binding precedent , yet it essentially created the doctrine of promissory estoppel .
</item 5>

<item 6>
Central London Property Trust Ltd v High Trees House Ltd
I am satisfied that a promise such as that to which I have referred is binding and the only question remaining for my consideration is the scope of the promise
in the present case . I am satisfied on all the evidence that the promise here was that the ground rent should be reduced to £ 1,250 a year as a temporary
expedient while the block of flats was not fully , or substantially fully let , owing to the conditions prevailing . That means that the reduction in the rent
applied throughout the years down to the end of 1944 , but early in 1945 it is plain that the flats were fully let , and , indeed the rents received from them (
many of them not being affected by the Rent Restrictions Acts ) , were increased beyond the figure at which it was originally contemplated that they would be
let . At all events the rent from them must have been very considerable . I find that the conditions prevailing at the time when the reduction in rent was made
, had completely passed away by the early months of 1945 .
</item 6>

<item 7>
Central London Property Trust Ltd v High Trees House Ltd
It was a representation , in effect , as to the future , namely , that payment of the rent would not be enforced at the full rate but only at the reduced rate .
Such a representation would not give rise to an estoppel , because , as was said in Jorden v. Money ( 1854 ) 5 H.L.C. 185 , a representation as to the future
must be embodied as a contract or be nothing . But what is the position in view of developments in the law in recent years ? The law has not been standing still
since Jorden v. Money ( 1854 ) 5 H.L.C. 185 . There has been a series of decisions over the last fifty years which , although they are said to be cases of
estoppel are not really such . They are cases in which a promise was made which was intended to create legal relations and which , to the knowledge of the
person making the promise , was going to be acted on by the person to whom it was made and which was in fact so acted on . In such cases the courts have said
that the promise must be honoured .
</item 7>

<item 8>
Central London Property Trust Ltd v High Trees House Ltd
Denning J held estoppel to be applicable if ... ... a promise was made which was intended to create legal relations and which , to the knowledge of the person
making the promise , was going to be acted on by the person to whom it was made and which was in fact so acted on . Contents ( hide ) 1 Facts 2 Judgment 3
Significance 4 See also 5 References Facts ( edit ) In 1937 , High Trees House Ltd leased a block of flats in Clapham , London , for a rate £ 2500 / year from
Central London Property Trust Ltd . Due to the conditions during the beginning of World War II occupancy rates were drastically lower than normal . In January
1940 , to ameliorate the situation , the parties made an agreement in writing to reduce rent by half . However , neither party stipulated the period for which
this reduced rental was to apply . Over the next five years , High Trees paid the reduced rate while the flats began to fill , and by 1945 , the flats were back
at full occupancy . Central London sued for payment of the full rental costs from June 1945 onwards ( i.e.
</item 8>

<item 9>
Central London Property Trust Ltd v High Trees House Ltd
Central London Property Trust Ltd v. High Trees House Ltd - wikipedia Central London Property Trust Ltd v. High Trees House Ltd Jump to : navigation , search
High Trees case Court King 's Bench Division Decided 18 July 1946 Citation ( s ) ( 1947 ) KB 130 ; ( 1956 ) 1 All ER 256 ; 62 TLR 557 ; ( 1947 ) LJR 77 ; 175 LT
333 Court membership Judge ( s ) sitting Denning J Keywords Promissory estoppel Central London Property Trust Ltd v High Trees House Ltd ( 1947 ) KB 130 ( or
the High Trees case ) is an English contract law decision in the High Court . It reaffirmed and extended the doctrine of promissory estoppel in contract law in
England and Wales . However , the most significant part of the judgment is obiter dicta as it relates to hypothetical facts ; that is , the landlord did not
seek repayment of the full wartime rent .
</item 9>

<item 10>
Central London Property Trust Ltd v High Trees House Ltd
Retrieved from `` https://en.wikipedia.org/w/index.php?title=Central_London_Property_Trust_Ltd_v_High_Trees_House_Ltd&oldid=837737647 '' Categories : English
contract case law Lord Denning cases English enforceability case law English estoppel case law 1946 in British law High Court of Justice cases 1946 in case law
Hidden categories : Use British English from August 2015 Use dmy dates from February 2012 Talk Contents About Wikipedia Add links This page was last edited on
22 April 2018 , at 18 : 40 . Text is available under the Creative Commons Attribution - ShareAlike License ; additional terms may apply . By using this site ,
you agree to the Terms of Use and Privacy Policy . Wikipedia ® is a registered trademark of the Wikimedia Foundation , Inc. , a non-profit organization . About
Wikipedia
</item 10>
=== END ITEM id=6fd49e8b71193d84 ===

=== ITEM id=691ed162b33b564c ===
Gold sentences and facts:
Sentence 0: Mount Baker - Snoqualmie National Forest and Nooksack Falls in the North Cascades range of , Washington , deer hunting scenes .
  Fact 0.0: Mount Baker-Snoqualmie National Forest is in the North Cascades range of Washington
  Fact 0.1: Nooksack Falls is in the North Cascades range of Washington
  Fact 0.2: Deer hunting scenes are associated with Mount Baker-Snoqualmie National Forest
  Fact 0.3: Deer hunting scenes are associated with Nooksack Falls
Sentence 1: St. Theodosius Russian Orthodox Cathedral , in the Tremont neighborhood of Cleveland , Ohio . The name plaque is clearly visible in one scene . Lemko Hall , Cleveland , Ohio . Also located in Tremont , the wedding banquet was filmed here . The name is clearly visible in one scene . U.S. Steel Central Furnaces in Cleveland , Ohio . Opening sequence steel mill scenes . Patpong , Bangkok , Thailand , the area used to represent Saigon 's red light district . Sai Yok , Kanchanaburi Province , Thailand Mount Baker - Snoqualmie National Forest and Nooksack Falls in the North Cascades range of , Washington , deer hunting scenes . Steubenville , Ohio , for some mill and neighborhood shots . Struthers , Ohio , for external house and long - range road shots . Also including , the town 's bowling alley is the Bowladrome Lanes , located at 56 State Street , Struthers , Ohio . Weirton , West Virginia , for mill and trailer shots . River Kwai , Thailand , Prison camp and initial Russian roulette scene .
  Fact 1.0: St. Theodosius Russian Orthodox Cathedral is in the Tremont neighborhood of Cleveland, Ohio
  Fact 1.1: The name plaque of St. Theodosius Russian Orthodox Cathedral is clearly visible in one scene
  Fact 1.2: Lemko Hall is in Cleveland, Ohio
  Fact 1.3: Lemko Hall is also located in Tremont
  Fact 1.4: The wedding banquet was filmed at Lemko Hall
  Fact 1.5: The name of Lemko Hall is clearly visible in one scene
  Fact 1.6: U.S. Steel Central Furnaces are in Cleveland, Ohio
  Fact 1.7: Opening sequence steel mill scenes are at U.S. Steel Central Furnaces
  Fact 1.8: Patpong, Bangkok, Thailand is the area used to represent Saigon's red light district
  Fact 1.9: Sai Yok is a location in Kanchanaburi Province, Thailand
  Fact 1.10: Mount Baker-Snoqualmie National Forest is in the North Cascades range of Washington
  Fact 1.11: Nooksack Falls is in the North Cascades range of Washington
  Fact 1.12: Deer hunting scenes are associated with Mount Baker-Snoqualmie National Forest
  Fact 1.13: Deer hunting scenes are associated with Nooksack Falls
  Fact 1.14: Steubenville, Ohio is a location used for some mill and neighborhood shots
  Fact 1.15: Struthers, Ohio is a location used for external house and long-range road shots
  Fact 1.16: The town's bowling alley is the Bowladrome Lanes
  Fact 1.17: The Bowladrome Lanes are located at 56 State Street, Struthers, Ohio
  Fact 1.18: Weirton, West Virginia is a location used for mill and trailer shots
  Fact 1.19: River Kwai, Thailand is the location of the prison camp and initial Russian roulette scene

Retrieved items:
<item 1>
The Deer Hunter
Struthers , Ohio , for external house and long - range road shots . Also including , the town 's bowling alley is the Bowladrome Lanes , located at 56 State
Street , Struthers , Ohio . Weirton , West Virginia , for mill and trailer shots . River Kwai , Thailand , Prison camp and initial Russian roulette scene .
Post-production ( edit ) By this point , The Deer Hunter had cost $13 million and the film still had to go through an arduous post-production . Film editor
Peter Zinner was given 600,000 feet of printed film to edit , a monumental task at the time . Producers Spikings and Deeley were pleased with the first cut ,
which ran for three and a half hours . `` We were thrilled by what we saw , '' wrote Deeley , `` and knew that within the three and a half hours we watched
there was a riveting film . '' Executives from Universal , including Lew Wasserman and Sid Sheinberg , were not very enthusiastic . `` I think they were shocked
, '' recalled Spikings . `` What really upset them was ' God Bless America ' .
</item 1>

<item 2>
The Deer Hunter
The reception scene was filmed at nearby Lemko Hall . The amateur extras lined up for the crowded wedding - dance sequences drank real liquor and beer . The
scenes were filmed in the summer , but were set in the fall . To accomplish a look of fall , individual leaves were removed from deciduous trees . Zsigmond also
had to desaturate the colors of the exterior shots , partly in camera and in the laboratory processing . The production manager asked each of the Russian
immigrant extras to bring to the location a gift - wrapped box to double for wedding presents . The manager figured if the extras did this , not only would the
production save time and money , but the gifts would also look more authentic . Once the unit wrapped and the extras disappeared , the crew discovered to their
amusement that the boxes were n't empty but filled with real presents , from china to silverware . `` Who got to keep all these wonderful offerings , '' wrote
Deeley `` is a mystery I never quite fathomed . '' Cimino originally claimed that the wedding scene would take up 21 minutes of screen time . In the end , it
took 51 minutes .
</item 2>

<item 3>
The Deer Hunter
The Deer Hunter - wikipedia The Deer Hunter Jump to : navigation , search For other uses , see Deer Hunter ( disambiguation ) . The Deer Hunter UK theatrical
release poster Directed by Michael Cimino Produced by Barry Spikings Michael Deeley Michael Cimino John Peverall Screenplay by Deric Washburn Story by Deric
Washburn Michael Cimino Louis Garfinkle Quinn K. Redeker Starring Robert De Niro John Cazale John Savage Meryl Streep Christopher Walken Music by Stanley Myers
Cinematography Vilmos Zsigmond Edited by Peter Zinner Production company EMI Distributed by Universal Pictures Release date December 8 , 1978 ( 1978 - 12 - 08 )
( Los Angeles ) February 23 , 1979 ( 1979 - 02 - 23 ) ( United States ) Running time 183 minutes Country United States Language English Russian Vietnamese
French Budget $15 million Box office $49 million The Deer Hunter is a 1978 American epic war drama film co-written and directed by Michael Cimino about a trio
of Russian American steelworkers whose lives are changed forever after they fight in the Vietnam War .
</item 3>

<item 4>
The Deer Hunter
His direction to his actors : `` You put the gun to your head , Chris , you shoot , you fall over and Bobby cradles your head . '' Filming locations ( edit )
Lemko Hall St. Theodosius Russian Orthodox Cathedral , in the Tremont neighborhood of Cleveland , Ohio . The name plaque is clearly visible in one scene . Lemko
Hall , Cleveland , Ohio . Also located in Tremont , the wedding banquet was filmed here . The name is clearly visible in one scene . U.S. Steel Central Furnaces
in Cleveland , Ohio . Opening sequence steel mill scenes . Patpong , Bangkok , Thailand , the area used to represent Saigon 's red light district . Sai Yok ,
Kanchanaburi Province , Thailand Mount Baker - Snoqualmie National Forest and Nooksack Falls in the North Cascades range of , Washington , deer hunting scenes .
Steubenville , Ohio , for some mill and neighborhood shots . Struthers , Ohio , for external house and long - range road shots .
</item 4>

<item 5>
The Deer Hunter
2 Washburn 's claim 3.2. 3 Deeley 's reaction to the revised script 4 Filming 4.1 The wedding scenes 4.2 The bar and the steel mill 4.3 Hunting the deer 4.4
Vietnam and the Russian roulette scenes 4.5 Filming locations 5 Post-production 5.1 Sound design 5.2 Previews 6 Soundtrack 6.1 Selected tracks 7 Release 8
Analysis 8.1 Controversy over Russian roulette 8.1. 1 Critics ' response 8.1.
</item 5>

<item 6>
The Deer Hunter
( Television Production ) . Chicago , IL : WTTW . ^ Jump up to : The Deer Hunter ( 1978 ) -- Filming locations Jump up ^ `` Lemko Hall '' . The Encyclopedia of
Cleveland History . May 24 , 2002 . Jump up ^ `` 7 Movie Locations in Bangkok - Bangkok.com Magazine '' . Bangkok.com . Retrieved 2016 - 10 - 17 . Jump up ^ Nam
, Suzanne ( 2012 ) . Moon Thailand . Avalon Travel . p. 320 . ISBN 9781598809695 . Jump up ^ Krause , Mariella ( 2013 ) . Lonely Planet Pacific Northwest 's
Best Trips . Lonely Planet . p. 118 . ISBN 1741798159 . Jump up ^ `` The Deer Hunter film locations ( 1978 ) '' . Movie-locations.com . Retrieved 2016 - 10 - 17
. Jump up ^ Randy Fox Writer and photographer ( 2012 - 05 - 05 ) .
</item 6>

<item 7>
The Deer Hunter
In the end , it took 51 minutes . Deeley believes that Cimino always planned to make this prologue last for an hour , and `` the plan was to be advanced by
stealth rather than straight dealing . '' At this point in the production , nearly halfway through principal photography , Cimino was already over budget , and
producer Spikings could tell from the script that shooting the extended scene could sink the project . The bar and the steel mill ( edit ) The bar was specially
constructed in an empty storefront in Mingo Junction , Ohio for $25,000 ; it later became an actual saloon for local steel mill workers . U.S. Steel allowed
filming inside its Cleveland mill , including placing the actors around the furnace floor , only after securing a $5 million insurance policy . Other filming
took place in Pittsburgh . Hunting the Deer ( edit ) The first deer to be shot was depicted in a `` gruesome close - up '' , although he was hit with a
tranquilizer dart . The stag that Michael allows to get away later was the same one used on TV commercials for the Connecticut Life Insurance Company .
</item 7>

<item 8>
Graph query row:
location: Nooksack Falls
scene: The Deer Hunter
film_scene_relationship: RELATED_TO
scene_location_relationship: RELATED_TO
</item 8>

<item 9>
The Deer Hunter
`` ' The Deer Hunter ' And ' Reckless ' : Hollywood Briefly Comes To The Factory Valleys ( PHOTOS ) '' . Huffington Post . Retrieved 2016 - 10 - 17 . Jump up ^
`` History - Struthers , Ohio '' . Strutherscommunity.weebly.com . Retrieved 2016 - 10 - 17 . Jump up ^ `` Youngstown News , Bowladrome Lanes owners reflect on
40 years in the business '' . Vindy.com . Retrieved 2016 - 10 - 17 . Jump up ^ Sterling , Joe ( July 5 , 1077 ) . `` '' Deer Hunter '' Filming Under Way `` .
Weirton Daily Times . West Virginia Division of Culture and History . Jump up ^ `` The Deer Hunter film locations '' . The Worldwide Guide To Movie Locations .
September 7 , 2015 . ^ Jump up to : McLellan , Dennis ( November 16 , 2007 ) . `` Peter Zinner , 88 ; film editor won Oscar for ' The Deer Hunter ' '' . Los
Angeles Times .
</item 9>

<item 10>
The Deer Hunter
The cast and crew viewed large amounts of news footage from the war to ensure authenticity . The film was shot over a period of six months . The Clairton scenes
comprise footage shot in eight different towns in four states : West Virginia , Pennsylvania , Washington , and Ohio . The initial budget of the film was $8.5
million . Meryl Streep accepted the role of the `` vague , stock girlfriend '' , in order to remain for the duration of filming with John Cazale , who had been
diagnosed with lung cancer . De Niro had spotted Streep in her stage production of The Cherry Orchard and had suggested that she play his girlfriend Linda .
Before the beginning of principal photography , Deeley had a meeting with the film 's appointed line producer Robert Relyea . Deeley hired Relyea after meeting
him on the set of Bullitt ( 1968 ) and was impressed with his experience . However , Relyea declined the job , refusing to disclose his reason why . Deeley
suspected that Relyea sensed in director Cimino something that would have made production difficult .
</item 10>
=== END ITEM id=691ed162b33b564c ===

=== ITEM id=dec27325a7996858 ===
Gold sentences and facts:
Sentence 0: A cover of the song by Stacey Markus is used at the end of episode nine ( `` The Invitation '' ) of the 2015 Netflix original series Grace and Frankie , starring Jane Fonda and Lily Tomlin .
  Fact 0.0: A cover of a song by Stacey Markus is used at the end of episode nine, 'The Invitation', of Grace and Frankie.
  Fact 0.1: Grace and Frankie is a 2015 Netflix original series.
  Fact 0.2: Grace and Frankie stars Jane Fonda.
  Fact 0.3: Grace and Frankie stars Lily Tomlin.

Retrieved items:
<item 1>
Let My Love Open the Door
Initially , Townshend 's manager despised the track due to it `` not sounding like Townshend , '' and wanted it to be removed from Empty Glass . However , upon
the song 's chart success , his manager called to apologize . Despite the song 's critical and commercial success , Pete Townshend did not consider it one of
his best songs . He told Rolling Stone in an interview that `` Let My Love Open the Door '' was `` just a ditty , '' also claiming that he preferred his minor
US hit `` A Little Is Enough '' from the same album . In 1996 , Townshend released a new version of the song , called `` the E. Cola mix '' , turning the song
into a ballad . This version appeared in different television shows and film soundtracks . Chart performance ( edit ) Weekly charts ( edit ) Chart ( 1980 ) Peak
position Canada RPM 100 5 UK Singles Chart 46 U.S. Billboard Hot 100 9 U.S. Cash Box Top 100 11 Year - end charts ( edit ) Chart ( 1980 ) Rank Canada 35 U.S.
</item 1>

<item 2>
Let My Love Open the Door
Let My Love Open the Door - wikipedia Let My Love Open the Door Jump to : navigation , search `` Let My Love Open the Door '' Single by Pete Townshend from the
album Empty Glass B - side `` Greyhound Girl '' Released June 4 , 1980 ( US ) Format 7 '' Recorded March 1980 Genre Art rock , new wave Length 2 : 44 Label Atco
Records Songwriter ( s ) Pete Townshend Producer ( s ) Pete Townshend ,
</item 2>

<item 3>
Let My Love Open the Door
The group Rogue Wave also covered it in `` Cover me '' released in 2017 . List of films , TV shows , and other media featuring the track ( edit ) The song has
been used frequently in film , most notably the comedy genre , often as trailer music for movies such as Jerry Maguire and How Do You Know , and then as end
credit song for Mr. Deeds in 2002 ; Jersey Girl and Along Came Polly , both in 2004 . It was featured in the following films and TV series : Look Who 's Talking
( 1989 ) Grosse Pointe Blank as `` E. Cola Mix '' ( 1997 ) Mr.
</item 3>

<item 4>
Let My Love Open the Door
Chris Thomas Pete Townshend UK singles chronology `` Rough Boys '' ( 1980 ) `` Let My Love Open the Door '' ( 1980 ) `` Keep on Working '' ( 1980 ) `` Rough
Boys '' ( 1980 ) `` Let My Love Open the Door '' ( 1980 ) `` Keep on Working '' ( 1980 ) Pete Townshend US singles chronology `` Keep Me Turning '' ( 1977 ) (
with Ronnie Lane ) Keep Me Turning1977 `` Let My Love Open the Door '' ( 1980 ) Let My Love Open the Door1980 `` A Little Is Enough '' ( 1980 ) A Little Is
Enough 1980 `` Let My Love Open the Door '' is a song written and performed by Pete Townshend from his 1980 album Empty Glass . It reached the top ten in the
United States in that same year , reaching number nine . It reached number five in Canada .
</item 4>

<item 5>
Let My Love Open the Door
at 22 : 30 . Text is available under the Creative Commons Attribution - ShareAlike License ; additional terms may apply . By using this site , you agree to the
Terms of Use and Privacy Policy . Wikipedia ® is a registered trademark of the Wikimedia Foundation , Inc. , a non-profit organization . About Wikipedia
</item 5>

<item 6>
Let My Love Open the Door
Imdb.com. 2017 - 05 - 12 . Retrieved 2018 - 03 - 30 . External links ( edit ) Lyrics of this song at MetroLyrics ( hide ) Pete Townshend Discography Studio
albums Who Came First Rough Mix ( w / Ronnie Lane ) Empty Glass All the Best Cowboys Have Chinese Eyes White City : A Novel The Iron Man : A Musical
Psychoderelict Classic Quadrophenia Meher Baba tribute albums ( w / Ronnie Lane et al . ) Happy Birthday I Am With Love O ' Parvardigar Live albums Deep End
Live !
</item 6>

<item 7>
Let My Love Open the Door
A cover of the song by Stacey Markus is used at the end of episode nine ( `` The Invitation '' ) of the 2015 Netflix original series Grace and Frankie ,
starring Jane Fonda and Lily Tomlin . The song is featured in a Walmart Christmas holiday television commercial which started airing in early December 2017 .
Lethal Weapon ( TV series ) , season 2 , episode 9 `` Fools Rush In '' play the song performed by `` Rogue wave '' ( 2017 ) . References ( edit ) Jump up ^
Unterberger , Richie ( 2012 - 06 - 14 ) . `` Let My Love Open the Door '' . Rolling Stone . Retrieved 2016 - 10 - 16 . Jump up ^ `` Pete Townshend , ' Empty
Glass ' 100 Best Albums of the Eighties '' . Rolling Stone . 1989 - 11 - 16 . Retrieved 2016 - 10 - 16 . ^ Jump up to : Bac-lac.gc.ca ^ Jump up to : `` Let My
Love Open the Door '' . Songfacts.com .
</item 7>

<item 8>
Let My Love Open the Door
Psychoderelict O Parvardigar Related articles Songs Songs Written Albums Peter Meaden The Who Deep End Tommy ( film ) The Who 's Tommy Tommy ( EP ) Tommy (
soundtrack ) The Boy Who Heard Music Quadrophenia ( musical ) Quadrophenia ( film ) Quadrophenia ( soundtrack ) The Lifehouse Method Horse 's Neck Who I Am Eel
Pie Publishing Eel Pie Studios Double O Cliff Townshend Emma Townshend Simon Townshend The Wick Ashdown House , Oxfordshire The Boathouse , Twickenham Chapel
House , Twickenham Retrieved from `` https://en.wikipedia.org/w/index.php?title=Let_My_Love_Open_the_Door&oldid=838422442 '' Categories : 1980 singles Debut
singles Pete Townshend songs Songs written by Pete Townshend 1980 songs Atco Records singles Song recordings produced by Pete Townshend British new wave songs
Hidden categories : Webarchive template wayback links Music infoboxes with deprecated parameters Articles with hAudio microformats Talk Contents About Wikipedia
Čeština Español Euskara Edit links This page was last edited on 26 April 2018 ,
</item 8>

<item 9>
Let My Love Open the Door
It reached number five in Canada . Contents ( hide ) 1 Background 2 Chart performance 2.1 Weekly charts 2.2 Year - end charts 3 Personnel 4 Covers 5 List of
Films , TV Shows , and Other Media featuring the track 6 References 7 External links Background ( edit ) Although Townshend is a devotee to the religious guru
Meher Baba , he claimed in the liner notes of his Anthology CD that `` Jesus sings '' on the track . `` Let My Love Open the Door '' was released as the second
single from Empty Glass in Britain , where it was backed with the non-album tracks `` Classified '' and `` Greyhound Girl . '' The song was a minor British hit
, reaching # 46 . The song saw more success when it was released as the debut single from Empty Glass in America , where the song reached # 9 . It was Pete
Townshend 's only solo top 10 hit on the Billboard Hot 100 , but The Who 's song `` I Can See For Miles , '' which was written by Townshend , reached the same
position on the chart 13 years earlier .
</item 9>

<item 10>
Let My Love Open the Door
Billboard Hot 100 59 U.S. Cashbox 92 Personnel ( edit ) Pete Townshend : vocals , guitars , synths John `` Rabbit '' Bundrick : `` straight '' keyboards Simon
Phillips : drums Tony Butler : bass guitar Ted Jensen - mastering engineer Covers ( edit ) A version of this song by Luminate appears on The Covers EP . It was
also covered by M. Ward ( for the 2005 Starbucks compilation album , Sweetheart 2005 : Love Songs ) Christian rock group Audio Adrenaline covered this song on
their 1999 album Underdog . Natalie Imbruglia covered the song on her album Male released in 2015 . Great Big Sea covered the song on their 2012 album XX .
Roger Daltrey covered the song in 2015 for a campaign involving Teen Cancer America and First Citizens Bank . A full version was released in February 2016 ,
with all proceeds going to Teen Cancer America . The group Rogue Wave also covered it in `` Cover me '' released in 2017 .
</item 10>
=== END ITEM id=dec27325a7996858 ===

=== ITEM id=5102fe9f6547a9da ===
Gold sentences and facts:
Sentence 0: The Compromise of 1877 was an informal , unwritten deal that settled the intensely disputed 1876 U.S. presidential election . It resulted in the United States federal government pulling the last troops out of the South , and formally ended the Reconstruction Era . Through the Compromise , Republican Rutherford B. Hayes was awarded the White House over Democrat Samuel J. Tilden on the understanding that Hayes would remove the federal troops whose support was essential for the survival of Republican state governments in South Carolina , Florida and Louisiana . The compromise involved Democrats who controlled the House of Representatives allowing the decision of the Electoral Commission to take effect . The outgoing president , Republican Ulysses S. Grant , removed the soldiers from Florida . As president , Hayes removed the remaining troops from South Carolina and Louisiana . As soon as the troops left , many white Republicans also left , and the `` Redeemer '' Democrats took control . They already dominated other state governments in the South . What was exactly agreed is somewhat contested as the documentation is insufficient .
  Fact 0.0: The Compromise of 1877 was an informal, unwritten deal.
  Fact 0.1: The Compromise of 1877 settled the intensely disputed 1876 U.S. presidential election.
  Fact 0.2: The Compromise resulted in the United States federal government pulling the last troops out of the South.
  Fact 0.3: The Compromise formally ended the Reconstruction Era.
  Fact 0.4: Through the Compromise, Republican Rutherford B. Hayes was awarded the White House over Democrat Samuel J. Tilden.
  Fact 0.5: The White House was awarded to Hayes on the understanding that Hayes would remove the federal troops.
  Fact 0.6: The federal troops' support was essential for the survival of Republican state governments in South Carolina, Florida and Louisiana.
  Fact 0.7: The compromise involved Democrats who controlled the House of Representatives.
  Fact 0.8: The compromise involved the Democrats allowing the decision of the Electoral Commission to take effect.
  Fact 0.9: The outgoing president, Republican Ulysses S. Grant, removed the soldiers from Florida.
  Fact 0.10: As president, Rutherford B. Hayes removed the remaining troops from South Carolina.
  Fact 0.11: As president, Rutherford B. Hayes removed the remaining troops from Louisiana.
  Fact 0.12: When the troops left, many white Republicans also left.
  Fact 0.13: When the troops left, the 'Redeemer' Democrats took control.
  Fact 0.14: The 'Redeemer' Democrats already dominated other state governments in the South.
  Fact 0.15: What was exactly agreed in the Compromise of 1877 is somewhat contested.
  Fact 0.16: The documentation of the Compromise of 1877 is insufficient.

Retrieved items:
<item 1>
Compromise of 1877
They met secretly at Wormley 's Hotel in Washington to forge a compromise with aid to internal improvements : bridges , canals and railroads wanted by the South
. But Peskin notes that no serious federal effort was made after Hayes took office to fund a railroad or provide other federal aid for improvements . An
opposing interest group representing the Southern Pacific actually thwarted Scott 's proposed Texas and Pacific scheme , and ultimately ran its own line to New
Orleans . Some historians , such as Allan Peskin , argue that the assurances offered to some Southern Democrats to prevent a filibuster were not a compromise
but a foregone conclusion , as Tilden did not command sufficient support . Peskin admits that Woodward 's interpretation had become almost universally accepted
in the nearly quarter century since he had published it . As not all terms of the agreement were met , Peskin believes there was really no deal between the
North and South in 1877 . He also suggests that Northern Democrats were more significant in quashing the filibuster than those from the South . For instance ,
Samuel J. Randall ( D - Pennsylvania ) was Speaker of the House and prevented the filibuster .
</item 1>

<item 2>
Compromise of 1877
The outgoing president , Republican Ulysses S. Grant , removed the soldiers from Florida . As president , Hayes removed the remaining troops from South Carolina
and Louisiana . As soon as the troops left , many white Republicans also left , and the `` Redeemer '' Democrats took control . They already dominated other
state governments in the South . What was exactly agreed is somewhat contested as the documentation is insufficient . Black Republicans felt betrayed as they
lost power and were subject to discrimination and harassment to suppress their voting . by 1905 most black men were effectively disenfranchised by state
legislatures in every southern state . Contents ( hide ) 1 Terms of compromise 2 Results 3 Interpretations 4 References 5 Further reading 6 External links Terms
of compromise ( edit ) The compromise essentially stated that Southern Democrats would acknowledge Hayes as president , but only on the understanding that
Republicans would meet certain demands . The following elements are generally said to be the points of the compromise : The removal of all U.S. military forces
from the former Confederate states . At the time , U.S.
</item 2>

<item 3>
Compromise of 1877
He tightened military security , and nobody marched on Washington . Hayes was peacefully inaugurated . Points 1 and 2 of the compromise took effect . Hayes had
already announced his support for the restoration of `` home rule , '' which would involve federal troop removal , before the election . It was not unusual ,
nor unexpected , for a president , especially one so narrowly elected , to select a cabinet member favored by the other party . Points 3 and 4 were never
enacted ; it is possible there was no firm agreement about them . Whether by informal deal or simply reassurances already in line with Hayes 's announced plans
, talks with Southern Democrats satisfied the worries of many . This prevented a Congressional filibuster that had threatened to extend resolution of the
election dispute beyond Inauguration Day 1877 . Interpretations ( edit ) C. Vann Woodward ( 1951 ) argued for an economic interpretation , saying that emerging
business and industry interests of the New South found common ground with Republican businessmen , particularly with the railroads .
</item 3>

<item 4>
Compromise of 1877
References ( edit ) Jump up ^ Michael Les Benedict , `` Southern Democrats in the Crisis of 1876 - 1877 : A Reconsideration of Reunion and Reaction . '' Journal
of Southern History ( 1980 ) : 489 - 524 . in JSTOR Jump up ^ Jones , Stephen A. ; Freedman , Eric ( 2011 ) . Presidents and Black America . CQ Press . p. 218 .
ISBN 9781608710089 . In an eleventh - hour compromise between party leaders - considered the `` Great Betrayal '' by many blacks and southern Republicans ...
Jump up ^ Woodward , C. Vann ( 1966 ) . Reunion and Reaction : The Compromise of 1877 and the End of Reconstruction . Boston : Little , Brown and Company . pp.
169 -- 171 . Jump up ^ Donald Richard Deskins ; Hanes Walton ; Sherman C. Puckett ( 2010 ) . Presidential Elections , 1789 - 2008 : County , State , and
National Mapping of Election Data . U of Michigan Press . p. 211 .
</item 4>

<item 5>
Compromise of 1877
military forces from the former Confederate states . At the time , U.S. troops remained in only Louisiana , South Carolina , and Florida , but the Compromise
completed their withdrawal from the region . The appointment of at least one Southern Democrat to Hayes ' cabinet . ( David M. Key of Tennessee was appointed as
Postmaster General . ) The construction of another transcontinental railroad using the Texas and Pacific in the South ( this had been part of the `` Scott Plan
, '' proposed by Thomas A. Scott of the Pennsylvania Railroad ; he had initiated negotiations resulting in the final compromise ) . Legislation to help
industrialize the South and restore its economy following Reconstruction and the Civil War . The right to deal with blacks without northern interference . In
exchange , Democrats would accept the Republican Hayes as president by not employing the filibuster during the joint session of Congress needed to confirm the
election . Results ( edit ) After the Compromise , a few Democrats complained loudly that Tilden had been cheated . There was talk of forming armed units that
would march on Washington , but President Grant was ready for that . He tightened military security , and nobody marched on Washington . Hayes was peacefully
inaugurated .
</item 5>

<item 6>
Compromise of 1877
( Registration required ( help ) ) . Polakoff , Keith Ian . The Politics of Inertia : The Election of 1876 and the End of Reconstruction ( 1973 ) Riddleberger ,
Patrick W. ( 1960 ) . `` The Radicals ' Abandonment of the Negro During Reconstruction '' . The Journal of Negro History . 45 ( 2 ) : 88 -- 102 . doi : 10.2307
/ 2716572 . JSTOR 2716572 -- via JSTOR . Simpson , Brooks D. `` Ulysses S. Grant and the Electoral Crisis of 1876 - 1877 , '' Hayes Historical Journal ( 1992 )
11 # 2 pp 5 -- 22 . Woodward , C. Vann ( 1951 ) . Reunion and Reaction : The Compromise of 1877 and the End of Reconstruction . Oxford University Press .
External links ( edit ) Samueltilden.com : History of Samuel J. Tilden website Rbhayes.org : R.B.
</item 6>

<item 7>
Compromise of 1877
Downs explores how Mexicanization was roundly rejected and stability was achieved . Whatever deals may or may not have taken place on the side , in formal legal
terms , the election of 1876 was not decided by such acts , but by the official vote of Congress to accept the recommendations of the Electoral Commission they
themselves had set up as a way out of the election impasse . The expectation in setting up the committee had been that its decisions would be accepted by
Congress . It was only when certain Democrats disagreed with the commission 's decisions in favor of Hayes that this arrangement was jeopardized . This
Democratic group threatened a filibuster ( opposed by Republicans and Congressional Democratic leadership as well ) that would prevent the agreed - upon vote
from taking place . Discussions of the points in the alleged compromise were related to persuading key Democrats against accepting a filibuster . The very
threat of a filibuster -- a measure used by a minority to prevent a vote -- indicates that there were already sufficient votes for accepting the commission 's
recommendations .
</item 7>

<item 8>
Compromise of 1877
( Oxford University Press , 1982 ) pp. 417 -- 50 . Downs , Gregory P. ( 2012 ) . `` The Mexicanization of American Politics : The United States ' Transnational
Path from Civil War to Stabilization '' . The American Historical Review . 117 ( 2 ) : 387 -- 409 . doi : 10.1086 / ahr. 117.2. 387 . Frantz , Edward O . The
Door of Hope : Republican Presidents and the First Southern Strategy , 1877 -- 1933 ( University Press of Florida . 2011 ) Peskin , Allan ( 1973 ) . `` Was
There a Compromise of 1877 '' . The Journal of American History . 60 ( 1 ) : 63 -- 75 . doi : 10.2307 / 2936329 . ISSN 1936 - 0967 . JSTOR 2936329 -- via JSTOR
. ( Registration required ( help ) ) . Polakoff , Keith Ian .
</item 8>

<item 9>
Compromise of 1877
Compromise of 1877 - wikipedia Compromise of 1877 Jump to : navigation , search A political cartoon by Joseph Keppler depicts Roscoe Conkling as a character
Mephistopheles ( the Devil ) while Rutherford B. Hayes strolls off with the prize of the `` Solid South '' depicted as a woman . The caption quotes Goethe : ``
Unto that Power he doth belong Which only doeth Right while ever willing Wrong . '' The Compromise of 1877 was an informal , unwritten deal that settled the
intensely disputed 1876 U.S. presidential election . It resulted in the United States federal government pulling the last troops out of the South , and formally
ended the Reconstruction Era . Through the Compromise , Republican Rutherford B. Hayes was awarded the White House over Democrat Samuel J. Tilden on the
understanding that Hayes would remove the federal troops whose support was essential for the survival of Republican state governments in South Carolina ,
Florida and Louisiana . The compromise involved Democrats who controlled the House of Representatives allowing the decision of the Electoral Commission to take
effect . The outgoing president , Republican Ulysses S. Grant , removed the soldiers from Florida .
</item 9>

<item 10>
Compromise of 1877
U of Michigan Press . p. 211 . Jump up ^ C. Vann Woodward ( 1991 ) . Reunion and Reaction : The Compromise of 1877 and the End of Reconstruction . Oxford UP .
pp. 200 -- 2 ` . Jump up ^ Downs , 2012 Jump up ^ C. Vann Woodward ( 1991 ) . Reunion and Reaction : The Compromise of 1877 and the End of Reconstruction .
Oxford UP . pp. 200 -- 2 ` . ^ Jump up to : Allan Peskin , `` Was There a Compromise of 1877 ? '' , The Journal of American History Vol. 60 , No. 1 ( Jun. ,
1973 ) , pp. 63 - 75 , via JSTOR Jump up ^ Vincent P. DeSantis , `` Rutherford B. Hayes and the Removal of the Troops and the End of Reconstruction , '' in J.
Morgan Kousser and James M. McPherson , eds.
</item 10>
=== END ITEM id=5102fe9f6547a9da ===

=== ITEM id=e8e36baa9ddcc1f9 ===
Gold sentences and facts:
Sentence 0: Kylo Ren is a fictional character in the Star Wars franchise . Introduced in the 2015 film Star Wars : The Force Awakens , he is portrayed by Adam Driver . Kylo Ren is the chosen name of Ben Solo , the son of original Star Wars trilogy characters Han Solo and Leia Organa . Though trained by his uncle Luke Skywalker as a Jedi , he has been seduced to the dark side of the Force by Supreme Leader Snoke and aspires to be as powerful as his grandfather , Darth Vader . Kylo Ren is also the master of the Knights of Ren , as well as a commander and later the supreme leader of the First Order , an organization spawned from the fallen Galactic Empire . He is featured in The Force Awakens media and merchandising and appears in the film 's sequel , Star Wars : The Last Jedi ( 2017 ) .
  Fact 0.0: Kylo Ren is a fictional character in the Star Wars franchise.
  Fact 0.1: Kylo Ren was introduced in the 2015 film Star Wars: The Force Awakens.
  Fact 0.2: Kylo Ren is portrayed by Adam Driver.
  Fact 0.3: Kylo Ren is the chosen name of Ben Solo.
  Fact 0.4: Ben Solo is the son of original Star Wars trilogy characters Han Solo and Leia Organa.
  Fact 0.5: Kylo Ren was trained by his uncle Luke Skywalker as a Jedi.
  Fact 0.6: Kylo Ren has been seduced to the dark side of the Force by Supreme Leader Snoke.
  Fact 0.7: Kylo Ren aspires to be as powerful as his grandfather, Darth Vader.
  Fact 0.8: Kylo Ren is the master of the Knights of Ren.
  Fact 0.9: Kylo Ren is a commander of the First Order.
  Fact 0.10: Kylo Ren later became the supreme leader of the First Order.
  Fact 0.11: The First Order is an organization spawned from the fallen Galactic Empire.
  Fact 0.12: Kylo Ren is featured in The Force Awakens media and merchandising.
  Fact 0.13: Kylo Ren appears in the film's sequel, Star Wars: The Last Jedi.
  Fact 0.14: Star Wars: The Last Jedi was released in 2017.
Sentence 1: Abrams told Empire in August 2015 , `` Kylo Ren is not a Sith . He works under Supreme Leader Snoke , who is a powerful figure on the Dark Side of the Force . '' Abrams had previously told Entertainment Weekly that the character , `` came to the name Kylo Ren when he joined a group called the Knights of Ren . '' Robbie Collin of The Telegraph described Ren as `` a hot - headed , radicalised Dark Side jihadi , whose red lightsaber splutters and crackles as violently as his temper '' . Abrams noted , `` The lightsaber is something that he built himself , and is as dangerous and as fierce and as ragged as the character . '' The Telegraph also explains that Ren 's wild and erratic temper and `` angsty '' instability make him dangerous . Melissa Leon of The Daily Beast describes Ren 's use of the Force as `` formidable '' , citing his ability to stop a blaster shot mid-air , immobilize victims and probe their minds against their will .
  Fact 1.0: Abrams told Empire in August 2015 that Kylo Ren is not a Sith.
  Fact 1.1: Abrams told Empire in August 2015 that Kylo Ren works under Supreme Leader Snoke.
  Fact 1.2: Abrams told Empire in August 2015 that Supreme Leader Snoke is a powerful figure on the Dark Side of the Force.
  Fact 1.3: Abrams had previously told Entertainment Weekly that the character came to the name Kylo Ren when he joined a group called the Knights of Ren.
  Fact 1.4: Robbie Collin of The Telegraph described Ren as a hot-headed, radicalised Dark Side jihadi.
  Fact 1.5: Robbie Collin of The Telegraph described Ren's red lightsaber as spluttering and crackling as violently as his temper.
  Fact 1.6: Abrams noted that the lightsaber is something Ren built himself.
  Fact 1.7: Abrams noted that the lightsaber is as dangerous, fierce and ragged as the character.
  Fact 1.8: The Telegraph explains that Ren's wild and erratic temper and angsty instability make him dangerous.
  Fact 1.9: Melissa Leon of The Daily Beast describes Ren's use of the Force as formidable.
  Fact 1.10: Melissa Leon cites Ren's ability to stop a blaster shot mid-air.
  Fact 1.11: Melissa Leon cites Ren's ability to immobilize victims.
  Fact 1.12: Melissa Leon cites Ren's ability to probe victims' minds against their will.

Retrieved items:
<item 1>
Kylo Ren
You normally think of order , and structure , and full commitment and no hesitation ... he 's just a little bit more unpolished . It 's in his costume , in his
lightsaber -- how you kind of get the sense that it could just not work at any moment ; that it could just blow up . That 's kind of like a big metaphor for him
. Appearances ( edit ) Model of Kylo 's cross-bladed lightsaber on display at Star Wars Launch Bay at Disney 's Hollywood Studios ( 2015 ) . Feature films (
edit ) The Force Awakens ( 2015 ) ( edit ) Main article : Star Wars : The Force Awakens Kylo Ren first appears in The Force Awakens as the commander of the
First Order , a sinister regime that has risen from the remains of the old Galactic Empire . After arriving at Jakku to retrieve a map containing the
coordinates for the location of Luke Skywalker , Ren kills an old vicar known as Lor San Tekka and captures Resistance pilot Poe Dameron , who has also been
sent to recover the map by General Leia Organa .
</item 1>

<item 2>
Kylo Ren
Ren 's character arc shares similarities with that of the Star Wars expanded universe character Jacen Solo , the son of Han Solo and Princess Leia who threatens
the galaxy as a fallen Jedi . Additionally , critics have noted a resemblance between Ren 's character design and that of Revan , the protagonist of Knights of
the Old Republic . Character ( edit ) Abrams told Empire in August 2015 , `` Kylo Ren is not a Sith . He works under Supreme Leader Snoke , who is a powerful
figure on the Dark Side of the Force . '' Abrams had previously told Entertainment Weekly that the character , `` came to the name Kylo Ren when he joined a
group called the Knights of Ren . '' Robbie Collin of The Telegraph described Ren as `` a hot - headed , radicalised Dark Side jihadi , whose red lightsaber
splutters and crackles as violently as his temper '' . Abrams noted , `` The lightsaber is something that he built himself , and is as dangerous and as fierce
and as ragged as the character . '' The Telegraph also explains that Ren 's wild and erratic temper and `` angsty '' instability make him dangerous .
</item 2>

<item 3>
Kylo Ren
Though trained by his uncle Luke Skywalker as a Jedi , he has been seduced to the dark side of the Force by Supreme Leader Snoke and aspires to be as powerful
as his grandfather , Darth Vader . Kylo Ren is also the master of the Knights of Ren , as well as a commander and later the supreme leader of the First Order ,
an organization spawned from the fallen Galactic Empire . He is featured in The Force Awakens media and merchandising and appears in the film 's sequel , Star
Wars : The Last Jedi ( 2017 ) . Contents ( hide ) 1 Concept and creation 2 Character 3 Appearances 3.1 Feature films 3.1. 1 The Force Awakens ( 2015 ) 3.1. 2
The Last Jedi ( 2017 ) 3.2 Related works and promotional material 3.3 Novels 4 Reception 5 Family tree 6 References 7 External links Concept and creation ( edit
) After Lucasfilm was acquired by The Walt Disney Company in 2012 , the script for The Force Awakens was written by Lawrence Kasdan , J.J.
</item 3>

<item 4>
Kylo Ren
Novels ( edit ) Kylo Ren is a point of view character in the 2015 novelization of The Force Awakens by Alan Dean Foster . In the 2017 Chuck Wendig novel Star
Wars : Aftermath : Empire 's End , Ben Solo is stated as having been born on the planet Chandrila on the same day as a peace treaty is signed between the
remnants of the Empire and the New Galactic Republic . Kylo Ren is additionally mentioned in the 2017 novel Star Wars : Phasma . Reception ( edit ) Adam Driver
received positive reviews for his performance as Kylo Ren . Both the character of Kylo Ren and Driver 's performance have been positively received . Many
complimented Ren 's conflicted nature and depth , as well as his costume design , and noted there were many places the character could be taken in future
installments . Peter Bradshaw of The Guardian praised the character and the actor alike , saying , `` He is gorgeously cruel , spiteful and capricious -- and
unlike the Vader of old , he is given to petulant temper tantrums , with his lightsaber drawn . ''
</item 4>

<item 5>
Kylo Ren
Abrams Michael Arndt Portrayed by Adam Driver Voiced by Adam Driver ( Disney Infinity 3.0 , Lego Star Wars : The Force Awakens , Lego Star Wars : The Resistance
Rises ) Matthew Wood ( Star Wars Battlefront II ) Roger Craig Smith ( Star Wars Battlefront II ) Information Full name Ben Solo Species Human Gender Male
Occupation Supreme Leader of the First Order Master of the Knights of Ren Former Jedi Padawan Affiliation First Order Knights of Ren New Jedi Order ( formerly )
Family Han Solo ( father ) Leia Organa ( mother ) Relatives Luke Skywalker ( maternal uncle ) Anakin Skywalker ( maternal grandfather ) Padmé Amidala ( maternal
grandmother ) Bail Organa ( adopted maternal grandfather ) Breha Organa ( adopted maternal grandmother ) Shmi Skywalker ( maternal great - grandmother )
Homeworld Chandrila Kylo Ren is a fictional character in the Star Wars franchise . Introduced in the 2015 film Star Wars : The Force Awakens , he is portrayed
by Adam Driver . Kylo Ren is the chosen name of Ben Solo , the son of original Star Wars trilogy characters Han Solo and Leia Organa .
</item 5>

<item 6>
Kylo Ren
Kylo Ren - wikipedia Kylo Ren Jump to : navigation , search `` Kylo '' redirects here . For the web browser , see Kylo ( web browser ) . Kylo Ren Ben Solo Star
Wars character Kylo Ren 's costume displayed on a mannequin First appearance The Force Awakens ( 2015 ) Created by Lawrence Kasdan J.J.
</item 6>

<item 7>
Kylo Ren
Before the duel is finished , they are separated by a seismic fissure created by the collapsing base . Rey and the others escape as Snoke orders General Hux to
evacuate the base and bring Ren to him to complete his training . The last Jedi ( 2017 ) ( edit ) Main article : Star Wars : The Last Jedi Ren 's inner conflict
continues into The Last Jedi , specifically through his conversations with Rey , with whom he connects through the Force . Rey believes that his conflict will
ultimately bring him back to the light side . Ren captures her and brings her before Snoke , who orders Ren to kill her . However , Ren uses the Force to ignite
a lightsaber at Snoke 's side and cut him in half , killing him . Ren beseeches Rey to join him in creating a new order , separate from the legacies created by
Snoke and Luke . When Rey refuses , Ren declares himself Supreme Leader of the First Order , and orders his forces to attack the Resistance base on Crait .
</item 7>

<item 8>
Kylo Ren
Jump up ^ Travers , Peter ( December 16 , 2015 ) . `` Star Wars : The Force Awakens '' . Rolling Stone . Retrieved December 21 , 2015 . Jump up ^ Buchanan ,
Kyle ( November 11 , 2015 ) . `` Let 's Discuss the Ending of Star Wars : The Force Awakens '' . Vulture.com . Retrieved December 19 , 2015 . Jump up ^ Cohen ,
David S. ( June 23 , 2016 ) . `` The Force Awakens Rings Up Eight Saturn Awards '' . Variety . Retrieved June 23 , 2016 . Jump up ^ Agar , Chris ( January 22 ,
2018 ) . `` Kylo Ren Voted Seventh - Greatest Movie Villain '' . Screen Rant . Retrieved January 24 , 2018 . External links ( edit ) Wikimedia Commons has media
related to Kylo Ren . Kylo Ren in the StarWars.com Databank Kylo Ren on Wookieepedia ,
</item 8>

<item 9>
Kylo Ren
Ren soon learns that the pilot had entrusted his astromech droid , BB - 8 , with the map . Poe flees with the help of rogue stormtrooper Finn , who later finds
BB - 8 , and the scavenger Rey . Finn , Rey , and BB - 8 escape Jakku in the Millennium Falcon , and are soon intercepted by the ship 's former owner , Han Solo
, and his co-pilot Chewbacca . It is revealed that Ren is the son of Han and Leia , and was once one of Luke 's Jedi pupils . He was drawn to the dark side of
the Force by the First Order 's Supreme Leader Snoke . He then helped to destroy his uncle Luke 's new Jedi Academy . However , Ren still feels the pull of the
light side of the Force and seeks the strength to overcome this weakness from his grandfather Darth Vader , whose burnt helmet is in Ren 's possession .
Arriving at the Maz Kanata castle , Ren captures Rey , who he senses has seen the map . While interrogating her with the Force , he realizes that she is strong
with the Force herself , though unaware of it .
</item 9>

<item 10>
Kylo Ren
Abrams and Michael Arndt , with no direct participation from Star Wars creator George Lucas . According to director Abrams , the First Order is inspired by
ODESSA , a theoretical network of SS officers who fled to Argentina following World War II . Driver 's casting in the film in an unnamed role was first
announced on April 29 , 2014 . Kylo Ren was first seen , but still not named , in the 88 - second The Force Awakens teaser trailer released by Lucasfilm on
November 28 , 2014 , wielding a jagged red lightsaber with a crossguard . The name Kylo Ren was revealed by Entertainment Weekly in a Lucasfilm - designed Topps
- style trading card mockup in December 2014 . A May 2015 Vanity Fair photo shoot by Annie Leibovitz confirmed that Driver would be portraying Kylo . In an
interview with Time , The Force Awakens costume designer Michael Kaplan said that Abrams requested that Ren 's mask be designed to be memorable for a child .
After numerous attempts to gain Abrams ' approval , the final design was selected .
</item 10>
=== END ITEM id=e8e36baa9ddcc1f9 ===

=== ITEM id=92ae0772c6397cfe ===
Gold sentences and facts:
Sentence 0: Life of Pi is a Canadian fantasy adventure novel by Yann Martel published in 2001 . The protagonist is Piscine Molitor `` Pi '' Patel , an Indian boy from Pondicherry who explores issues of spirituality and practicality from an early age . He survives 227 days after a shipwreck while stranded on a lifeboat in the Pacific Ocean with a Bengal tiger named Richard Parker .
  Fact 0.0: Life of Pi is a Canadian fantasy adventure novel.
  Fact 0.1: Life of Pi is a novel by Yann Martel.
  Fact 0.2: Life of Pi was published in 2001.
  Fact 0.3: The protagonist of Life of Pi is Piscine Molitor 'Pi' Patel.
  Fact 0.4: Piscine Molitor 'Pi' Patel is an Indian boy from Pondicherry.
  Fact 0.5: Piscine Molitor 'Pi' Patel explores issues of spirituality and practicality from an early age.
  Fact 0.6: Pi survives 227 days after a shipwreck.
  Fact 0.7: Pi is stranded on a lifeboat in the Pacific Ocean.
  Fact 0.8: Pi is stranded on the lifeboat with a Bengal tiger named Richard Parker.

Retrieved items:
<item 1>
Life of Pi
Canongate Books . 14 May 2002 . Archived from the original on 18 March 2008 . Retrieved 1 September 2010 . Jump up ^ Martel , Yann . `` How Richard Parker Came
to Get His Name '' . Amazon.com . Retrieved 1 September 2010 . Jump up ^ `` From the Author -- Yann Martel -- Powell 's Books '' . Powells.com . Archived from
the original on 14 January 2013 . Retrieved 30 December 2012 . Jump up ^ Mitgang , Herbert ( 11 July 1990 ) . `` Books of The Times ; Fleeing the Nazis With a
Jaguar That May Be Real '' . The New York Times . Retrieved 2 September 2010 . Jump up ^ Rohter , Larry ( 11 July 1990 ) . `` Tiger in a Lifeboat , Panther in a
Lifeboat : A Furor Over a Novel '' . The New York Times . Retrieved 2 September 2010 .
</item 1>

<item 2>
Life of Pi
The cook amputates the sailor 's leg for use as fishing bait , then kills the sailor himself as well as Pi 's mother for food , and soon he is killed by Pi ,
who dines on him . The investigators note parallels between the two stories . They soon conclude that the hyena symbolizes the cook , the zebra the sailor , the
orangutan Pi 's mother , and the tiger represents Pi . Pi points out that neither story can be proven and neither explains the cause of the shipwreck , so he
asks the officials which story they prefer : the one without animals or the one with animals . They eventually choose the story with the animals . Pi thanks
them and says : `` And so it goes with God . '' The investigators then leave and file a report . Major themes ( edit ) Life is a story ( edit ) Life of Pi ,
according to Yann Martel , can be summarized in three statements : `` Life is a story ... You can choose your story ... A story with God is the better story .
'' A recurring theme throughout the novel seems to be believability .
</item 2>

<item 3>
Life of Pi
His father owns a zoo in Pondicherry . The livelihood provides the family with a relatively affluent lifestyle and some understanding of animal psychology . The
narrator describes how he acquired his full name , Piscine Molitor Patel , as a tribute to the swimming pool in France . After hearing schoolmates tease him by
transforming the first name into `` Pissing '' , he establishes the short form of his name as `` Pi '' when he starts secondary school . The name , he says ,
pays tribute to the irrational number which is the ratio of the circumference of a circle to its diameter . In recounting his experiences , Pi describes several
other unusual situations involving proper names : two visitors to the zoo , one a devout Muslim , and the other a committed atheist , bear identical names ; and
a 450 - pound tiger at the zoo bears the name Richard Parker as the result of a clerical error , in which human and animal names were reversed . Pi is raised as
a Hindu who practices vegetarianism . At the age of fourteen , he investigates Christianity and Islam , and decides to become an adherent of all three religions
, much to his parents ' dismay , saying he `` just wants to love God . ''
</item 3>

<item 4>
Life of Pi
He tries to understand God through the lens of each religion , and comes to recognize benefits in each one . A few years later in 1977 , during the period when
Indian Prime Minister Indira Gandhi declares `` The Emergency '' , Pi 's father decides to sell the zoo and immigrate with his wife and sons to Canada . Part
two ( edit ) The second part of the novel begins with Pi 's family aboard the Tsimtsum , a Japanese freighter that is transporting animals from their zoo to
North America . A few days out of port from Manila , the ship encounters a storm and sinks . Pi manages to escape in a small lifeboat , only to learn that the
boat also holds a spotted hyena , an injured Grant 's zebra , and an orangutan named Orange Juice . Much to the boy 's distress , the hyena kills the zebra and
then Orange Juice . A tiger has been hiding under the boat 's tarpaulin : it 's Richard Parker , who had boarded the lifeboat with ambivalent assistance from Pi
himself some time before the hyena attack . Suddenly emerging from his hideaway , Richard Parker kills and eats the hyena .
</item 4>

<item 5>
Life of Pi
Parker names the cub Thirsty after his enthusiasm when drinking from a nearby river . The paperwork that accompanies the shipment of the two tigers to Pi 's
family 's zoo in Pondicherry states that the cub 's name is `` Richard Parker '' and the hunter 's given name is `` Thirsty '' and his surname is `` None Given
'' , due to a mix - up with the names . Pi 's father finds the story so amusing that they continue to call the tiger `` Richard Parker '' . Reception ( edit )
Brian Bethune of Maclean 's describes Life of Pi as a `` head - scratching combination of dense religious allegory , zoological lore and enthralling adventure
tale , written with warmth and grace '' . Master Plots suggested that the `` central themes of Life of Pi concern religion and human faith in God '' . Reutter
said , `` So believable is Pi 's story telling that readers will be amazed . '' Gregory Stephens added that it `` achieves something more quietly spectacular .
'' Smith stated that there was `` no bamboozlement here . ''
</item 5>

<item 6>
Life of Pi
He was named after a swimming pool in Paris , despite the fact that neither his mother nor his father particularly liked swimming . The story is told as a
narrative from the perspective of a middle - aged Pi , now married with his own family , and living in Canada . At the time of main events of the story , he is
sixteen years old . He recounts the story of his life and his 227 - day journey on a lifeboat when his ship sinks in the middle of the Pacific Ocean during a
voyage to North America . Richard Parker ( edit ) A Bengal tiger Richard Parker is a royal Bengal tiger who is stranded on the lifeboat with Pi when the ship
sinks . Richard Parker lives on the lifeboat with Pi and is kept alive with the food and water Pi delivers . Richard Parker develops a relationship with Pi that
allows them to coexist in their struggle . In the novel , a hunter named Richard Parker is hired to kill a panther thought to have killed seven people within
two months . Instead , he accidentally immobilizes a female Bengal tiger with tranquilizer darts while her cub escapes hiding in a bush . Parker names the cub
Thirsty after his enthusiasm when drinking from a nearby river .
</item 6>

<item 7>
Life of Pi
Suddenly emerging from his hideaway , Richard Parker kills and eats the hyena . Frightened , Pi constructs a small raft out of rescue flotation devices ,
tethers it to the bow of the boat and makes it his place of retirement . He begins conditioning Richard Parker to take a submissive role by using food as a
positive reinforcer , and seasickness as a punishment mechanism , while using a whistle for signals . Soon , Pi asserts himself as the alpha animal , and is
eventually able to share the boat with his feline companion , admitting in the end that Richard Parker is the one who helped him survive his ordeal . Pi
recounts various events while adrift in the Pacific Ocean . At his lowest point , exposure renders him blind and unable to catch fish . In a state of delirium ,
he talks with a marine `` echo '' , which he initially identifies as Richard Parker having gained the ability to speak , but it turns out to be another blind
castaway , a Frenchman , who boards the lifeboat with the intention of killing and eating Pi , but is eventually killed by Richard Parker .
</item 7>

<item 8>
Life of Pi
Life of Pi - Wikipedia Life of Pi Jump to : navigation , search This article is about the novel by Yann Martel . For the film based on the novel and directed by
Ang Lee , see Life of Pi ( film ) . Life of Pi Life of Pi cover Author Yann Martel Original title Life of Pi Country Canada Language English Genre Fiction
Publisher Knopf Canada Publication date September 11 , 2001 ( Canada ) ISBN 0 - 676 - 97376 - 0 ( first edition , hardcover ) OCLC 46624335 Preceded by Self
Followed by Beatrice and Virgil Life of Pi is a Canadian fantasy adventure novel by Yann Martel published in 2001 . The protagonist is Piscine Molitor `` Pi ''
Patel , an Indian boy from Pondicherry who explores issues of spirituality and practicality from an early age . He survives 227 days after a shipwreck while
stranded on a lifeboat in the Pacific Ocean with a Bengal tiger named Richard Parker . The novel has sold more than ten million copies worldwide .
</item 8>

<item 9>
Life of Pi
A recurring theme throughout the novel seems to be believability . Pi at the end of the book asks the two investigators `` If you stumble at mere believability
, what are you living for ? '' According to Gordon Houser there are two main themes of the book : `` that all life is interdependent , and that we live and
breathe via belief . '' Growth through adversity ( edit ) PBS has described Martel 's story as one of `` personal growth through adversity . '' The main
character learns that `` tigers are dangerous '' at a young age when his father forces him to watch the zoo 's Royal Bengal tiger patriarch , Mahisha , devour a
live goat . Later , after he has been reduced to eking out a desperate existence on the lifeboat with the company of a fully grown tiger , Pi develops `` alpha
'' qualities as he musters the strength , will and skills he needs to survive . Inspiration ( edit ) In a 2002 interview with PBS , Martel said `` I was sort of
looking for a story , not only with a small ' s ' but sort of with a capital ' S ' -- something that would direct my life . ''
</item 9>

<item 10>
Life of Pi
He spoke of being lonely and needing direction in his life , and found that writing the novel met this need . Richard Parker and shipwreck narratives ( edit )
The name of Martel 's tiger , Richard Parker , was inspired by a character in Edgar Allan Poe 's nautical adventure novel The Narrative of Arthur Gordon Pym of
Nantucket ( 1838 ) . In this book , Richard Parker is a mutineer who is stranded and eventually cannibalized on the hull of an overturned ship ( and there is a
dog aboard who is named Tiger ) . The author also had in mind another occurrence of the name , in the famous legal case R v Dudley and Stephens ( 1884 ) where a
shipwreck again results in the cannibalism of a cabin boy named Richard Parker , this time in a lifeboat . A third Richard Parker drowned in the sinking of the
Francis Spaight in 1846 , described by author Jack London , and later the cabin boy ( not Richard Parker ) was cannibalized . Having read about these events ,
Yann Martel thought , `` So many victimized Richard Parkers had to mean something . ''
</item 10>
=== END ITEM id=92ae0772c6397cfe ===
