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

=== ITEM id=baf73ec5b3ca2502 ===
Gold sentences and facts:
Sentence 0: Songwriter ( s ) Kirsty MacColl / Philip Rambow
  Fact 0.0: The songwriters of the song are Kirsty MacColl and Philip Rambow.
Sentence 1: `` There 's a Guy Works Down the Chip Shop , Swears He 's Elvis '' Single by Kirsty MacColl from the album Desperate Character A-side `` There 's a Guy Works Down the Chip Shop Swears He 's Elvis '' B - side `` Hard To Believe There 's a Guy ... ( country version ) '' Released May 1981 Format 7 '' Vinyl Genre Country pop Label Polydor Songwriter ( s ) Kirsty MacColl / Philip Rambow Producer ( s ) Barry Farmer Kirsty MacColl singles chronology `` Keep Your Hands Off My Baby '' ( 1981 ) `` There 's a Guy Works Down the Chip Shop , Swears He 's Elvis '' ( 1981 ) `` See That Girl '' ( 1981 ) `` Keep Your Hands Off My Baby '' ( 1981 ) `` There 's a Guy Works Down the Chip Shop Swears He 's Elvis '' ( 1981 ) `` See That Girl '' ( 1981 )
  Fact 1.0: 'There's a Guy Works Down the Chip Shop, Swears He's Elvis' is a single by Kirsty MacColl.
  Fact 1.1: 'There's a Guy Works Down the Chip Shop, Swears He's Elvis' is from the album Desperate Character.
  Fact 1.2: The single's A-side is 'There's a Guy Works Down the Chip Shop Swears He's Elvis'.
  Fact 1.3: The single's B-side is 'Hard To Believe There's a Guy ... (country version)'.
  Fact 1.4: The single was released in May 1981.
  Fact 1.5: The single's format is 7-inch vinyl.
  Fact 1.6: The single's genre is country pop.
  Fact 1.7: The single's label is Polydor.
  Fact 1.8: The single's songwriters are Kirsty MacColl and Philip Rambow.
  Fact 1.9: The single's producer is Barry Farmer.
  Fact 1.10: The Kirsty MacColl singles chronology includes 'Keep Your Hands Off My Baby' (1981).
  Fact 1.11: The Kirsty MacColl singles chronology includes 'There's a Guy Works Down the Chip Shop, Swears He's Elvis' (1981).
  Fact 1.12: The Kirsty MacColl singles chronology includes 'See That Girl' (1981).
  Fact 1.13: The Kirsty MacColl singles chronology includes 'Keep Your Hands Off My Baby' (1981) a second time.
  Fact 1.14: The Kirsty MacColl singles chronology includes 'There's a Guy Works Down the Chip Shop Swears He's Elvis' (1981).
  Fact 1.15: The Kirsty MacColl singles chronology includes 'See That Girl' (1981) a second time.

Retrieved items:
<item 1>
The Rolling Stones
Jump up ^ Haslam 2015 , p. 91 . Jump up ^ Wyman 2002 , pp. 65 . Jump up ^ Wyman 2002 , pp. 80 -- 83 . Jump up ^ Fricke , David ( 17 April 2008 ) . `` Blues
Brothers '' . Rolling Stone . Archived from the original on 5 April 2008 . Jump up ^ Gilliland 1969 , show 30 , track 2 . Jump up ^ `` Yes , The Beatles Once
Wrote a Song for the Rolling Stones '' . Ultimate Classic Rock . Archived from the original on 9 July 2017 . Retrieved 15 August 2017 . Jump up ^ `` How The
Beatles ended up writing `` I Wanna Be Your Man '' for The Rolling Stones `` . The Beatles . Archived from the original on 15 August 2017 . Retrieved 15 August
2017 . ^ Jump up to : `` UK Charts -- The Rolling Stones '' . Official Charts Company . Archived from the original on 20 October 2014 .
</item 1>

<item 2>
There's a Guy Works Down the Chip Shop Swears He's Elvis
There 's a Guy Works Down the Chip Shop Swears He 's Elvis - wikipedia There 's a Guy Works Down the Chip Shop Swears He 's Elvis Jump to : navigation , search
This article relies too much on references to primary sources . Please improve this by adding secondary or tertiary sources . ( March 2008 ) ( Learn how and
when to remove this template message ) `` There 's a Guy Works Down the Chip Shop , Swears He 's Elvis '' Single by Kirsty MacColl from the album Desperate
Character A-side `` There 's a Guy Works Down the Chip Shop Swears He 's Elvis '' B - side `` Hard To Believe There 's a Guy ... ( country version ) '' Released
May 1981 Format 7 '' Vinyl Genre Country pop Label Polydor Songwriter ( s ) Kirsty MacColl / Philip Rambow Producer ( s ) Barry Farmer Kirsty MacColl singles
chronology `` Keep Your Hands Off My Baby '' ( 1981 ) `` There 's a Guy Works Down the Chip Shop ,
</item 2>

<item 3>
Graph query row:
song: There's a Guy Works Down the Chip Shop Swears He's Elvis
writer: P. Rambow
</item 3>

<item 4>
There's a Guy Works Down the Chip Shop Swears He's Elvis
`` Walking Down Madison '' `` Soho Square '' With The Pogues `` Fairytale of New York '' Related articles Ewan MacColl Steve Lillywhite Retrieved from ``
https://en.wikipedia.org/w/index.php?title=There%27s_a_Guy_Works_Down_the_Chip_Shop_Swears_He%27s_Elvis&oldid=799608311 '' Categories : 1981 singles 1981 songs
Elisabeth Andreassen songs Kirsty MacColl songs Songs written by Kirsty MacColl Songs about Elvis Presley Polydor Records singles Hidden categories : Articles
lacking reliable references from March 2008 All articles lacking reliable references Talk About Wikipedia Svenska Edit links This page was last edited on 8
September 2017 , at 19 : 12 . About Wikipedia
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
There's a Guy Works Down the Chip Shop Swears He's Elvis
Retrieved 2014 - 10 - 30 . Jump up ^ Information at Svensk mediedatabas External links ( edit ) freeworld - Kirsty MacColl fansite Kirsty MacColl Studio albums
Desperate Character Kirsty MacColl Kite Electric Landlady Titanic Days Tropical Brainstorm Compilations Galore Notable singles `` They Do n't Know '' `` Keep
Your Hands Off My Baby '' `` There 's a Guy Works Down the Chip Shop Swears He 's Elvis '' `` You Still Believe in Me '' `` Terry '' `` A New England '' `` Days
'' `` Do n't Come The Cowboy With Me Sonny Jim ! ''
</item 6>

<item 7>
There's a Guy Works Down the Chip Shop Swears He's Elvis
Swears He 's Elvis '' ( 1981 ) `` See That Girl '' ( 1981 ) `` Keep Your Hands Off My Baby '' ( 1981 ) `` There 's a Guy Works Down the Chip Shop Swears He 's
Elvis '' ( 1981 ) `` See That Girl '' ( 1981 ) `` There 's a Guy Works Down the Chip Shop Swears He 's Elvis '' was Kirsty MacColl 's first hit single ,
charting in the UK at # 14 in 1981 , and remaining in the charts for nine weeks . It reached # 9 in the Irish charts . A promotional version was released in the
USA , which instead referenced a truck stop as opposed to the very British chip shop , but this was only available to DJs and did not get a full release .
Norwegian singer Elisabeth Andreasson covered the song on her 1981 country album Angel of the Morning , with lyrics in Swedish by Hasse Olsson as `` Killen ner
' på Konsum svär att han är Elvis '' ( `` The guy down Konsum '' swears he 's Elvis ) .
</item 7>

<item 8>
Graph query row:
song: There's a Guy Works Down the Chip Shop Swears He's Elvis
writer: Kirsty MacColl
</item 8>

<item 9>
There's a Guy Works Down the Chip Shop Swears He's Elvis
Track listing ( edit ) `` There 's a Guy Works Down the Chip Shop Swears He 's Elvis '' ( K. MacColl / P. Rambow ) `` Hard to Believe '' ( K. MacColl ) `` There
's a Guy Works Down the Chip Shop Swears He 's Elvis ( country version ) '' ( K. MacColl / P. Rambow ) References ( edit ) Jump up ^ `` There 's a Guy Works
Down the Chip Shop Swears He 's Elvis - Kirsty MacColl '' . Retrieved 2014 - 10 - 30 . Jump up ^ `` There 's a Guy Works Down the Chip Shop Swears He 's Elvis 7
'' single - Kirsty MacColl `` . Retrieved 2014 - 10 - 30 .
</item 9>

<item 10>
All I Want for Christmas Is My Two Front Teeth
The song was introduced in 1948 by singing group The Satisfiers on Perry Como 's radio show , and originally recorded by Spike Jones & His City Slickers on
December 6 , 1947 , with lead vocal by George Rock . That version reached the top of the pop charts in 1948 , and again in 1949 . The song has also been
recorded by numerous other singers and performers , including Ray Stevens , George Strait , Danny Kaye with The Andrews Sisters , Urbie Green , The Platters ,
Dread Zeppelin , The Kelly Family , Nat King Cole ( reportedly Gardner 's favorite version ) , The Chipmunks , the Hampton String Quartet , The Three Stooges
and Count von Count of Sesame Street . In the versions by Spike , The Chipmunks ( Theodore on lead vocals ) , and Ray Stevens , the song ends with the performer
shouting the words `` Happy New Year ! ! ! '' Homer and Jethro parodied the song as `` All I Want for Christmas Is My Upper Plate '' , on their 1968 album Cool
Crazy Christmas .
</item 10>
=== END ITEM id=baf73ec5b3ca2502 ===

=== ITEM id=7ce67b12b2872c78 ===
Gold sentences and facts:
Sentence 0: The Partridge Family Genre Musical sitcom Created by Bernard Slade Starring Shirley Jones David Cassidy Danny Bonaduce Susan Dey Suzanne Crough Jeremy Gelbwaks Dave Madden Brian Forster Theme music composer Diane Hilderbrand Danny Janssen Wes Farrell Opening theme `` When We 're Singin ' '' ( 1970 -- 71 ) `` C'mon , Get Happy '' ( 1971 -- 74 ) Composer ( s ) George Duning Benny Golson Warren Barker Hugo Montenegro Shorty Rogers Country of origin United States Original language ( s ) English No. of seasons No. of episodes 96 ( list of episodes ) Production Executive producer ( s ) Bob Claver Producer ( s ) William Bickley Paul Junger Witt Dale McRaven Larry Rosen Mel Swope Cinematography Fred Jackman , Jr . Irving Lippman Camera setup Single - camera Running time 25 minutes Production company ( s ) Screen Gems Television ( 1970 -- 1974 ) Distributor Columbia Pictures Television ( 1974 -- 84 ; 1989 -- 96 ) DFS Program Exchange ( 1984 -- 87 ) The Program Exchange ( 1987 -- 89 ) Columbia TriStar Television ( 1996 -- 2002 ) Sony Pictures Television ( 2002 -- present ) Release Original network ABC Audio format Monaural Original release September 25 , 1970 ( 1970 - 09 - 25 ) -- March 23 , 1974 ( 1974 - 03 - 23 ) Chronology Related shows Getting Together Goober and the Ghost Chasers Partridge Family 2200 A.D.
  Fact 0.0: The Partridge Family is a musical sitcom
  Fact 0.1: The Partridge Family was created by Bernard Slade
  Fact 0.2: The Partridge Family starred Shirley Jones
  Fact 0.3: The Partridge Family starred David Cassidy
  Fact 0.4: The Partridge Family starred Danny Bonaduce
  Fact 0.5: The Partridge Family starred Susan Dey
  Fact 0.6: The Partridge Family starred Suzanne Crough
  Fact 0.7: The Partridge Family starred Jeremy Gelbwaks
  Fact 0.8: The Partridge Family starred Dave Madden
  Fact 0.9: The Partridge Family starred Brian Forster
  Fact 0.10: The Partridge Family's theme music composers were Diane Hilderbrand, Danny Janssen and Wes Farrell
  Fact 0.11: The Partridge Family's opening theme was 'When We're Singin'' (1970-71)
  Fact 0.12: The Partridge Family's opening theme was 'C'mon, Get Happy' (1971-74)
  Fact 0.13: The Partridge Family's composers were George Duning, Benny Golson, Warren Barker, Hugo Montenegro and Shorty Rogers
  Fact 0.14: The Partridge Family is from the United States
  Fact 0.15: The Partridge Family's original language is English
  Fact 0.16: The Partridge Family has 96 episodes
  Fact 0.17: The Partridge Family's executive producer was Bob Claver
  Fact 0.18: The Partridge Family's producers were William Bickley, Paul Junger Witt, Dale McRaven, Larry Rosen and Mel Swope
  Fact 0.19: The Partridge Family's cinematography was by Fred Jackman, Jr. and Irving Lippman
  Fact 0.20: The Partridge Family uses a single-camera setup
  Fact 0.21: The Partridge Family has a running time of 25 minutes
  Fact 0.22: The Partridge Family was produced by Screen Gems Television (1970-1974)
  Fact 0.23: The Partridge Family was distributed by Columbia Pictures Television (1974-84; 1989-96)
  Fact 0.24: The Partridge Family was distributed by DFS Program Exchange (1984-87)
  Fact 0.25: The Partridge Family was distributed by The Program Exchange (1987-89)
  Fact 0.26: The Partridge Family was distributed by Columbia TriStar Television (1996-2002)
  Fact 0.27: The Partridge Family was distributed by Sony Pictures Television (2002-present)
  Fact 0.28: The Partridge Family aired originally on ABC
  Fact 0.29: The Partridge Family has a monaural audio format
  Fact 0.30: The Partridge Family originally aired from September 25, 1970 to March 23, 1974
  Fact 0.31: Related shows include Getting Together
  Fact 0.32: Related shows include Goober and the Ghost Chasers
  Fact 0.33: Related shows include Partridge Family 2200 A.D.

Retrieved items:
<item 1>
The Partridge Family entry lists its cast—David Cassidy, Shirley Jones, Danny Bonaduce, Susan Dey, Jeremy Gelbwaks, Brian Forster, Suzanne Crough, and Dave
Madden—along with studio and compilation albums, notable singles such as “I Think I Love You,” related articles and spin-offs, and its MusicBrainz identifier.
</item 1>

<item 2>
*The Partridge Family* used two theme songs. Season 1 featured “When We’re Singin’,” whose lyrics describe the family helping their mother by making music, with
Danny getting Reuben to sell their song. In later seasons, it was replaced by “C’mon Get Happy.” Under Farrell’s direction, the new song used the original tune
with revised lyrics, including a verse about traveling together and spreading love.
</item 2>

<item 3>
The Partridge Family
More than one song was used as the theme song . Season 1 episodes feature the song `` When We 're Singin ' '' , which was replaced in subsequent seasons with ``
C'mon Get Happy '' . `` Five of us , and Mom working all day , we knew we could help her if our music would pay , Danny got Reuben to sell our song , and it
really came together when Mom sang along ... '' ( from `` When We 're Singin ' '' ) Later , under Farrell 's direction , the new version had new lyrics sung to
the `` When We 're Singin ' '' tune . With the new chorus finalized , `` C'mon Get Happy '' showcased the new verse : `` We had a dream , we 'd go travelin '
together , We 'd spread a little lovin ' then we 'd keep movin ' on .
</item 3>

<item 4>
The Partridge Family
The company promoted the show by releasing a series of albums featuring the family band , though David Cassidy and Shirley Jones , who sang background , were
the only cast members who were actually featured on the recordings . As the show and other associated merchandising soared , David Cassidy became a teen idol .
The producers signed Cassidy as a solo act as well . Cassidy began touring with his own group of musicians , performing Partridge songs as well as hits from his
own albums , to thousands of screaming teenagers in major stadiums across the USA , UK , Europe , Japan and Australia . The Partridge Family 's biggest hit came
in 1970 with the song `` I Think I Love You '' , written by Tony Romeo ( who had previously written several of the Cowsills ' hits ) , peaked at Number 1 on the
Billboard charts in December of that year . It sold over five million copies , outselling the Beatles ' `` Let It Be '' , was awarded a gold disc , and made the
group the third fictional artist to have a No. 1 hit ( after The Chipmunks and The Archies ) .
</item 4>

<item 5>
*The Partridge Family* was an American musical sitcom that aired on ABC from 1970 to 1974. Set in California, it centered on a fictional family and musical
group. The series was associated with Bell Records, based on singers and musicians, and later adapted into comics.
</item 5>

<item 6>
The Partridge Family
The Partridge Family - wikipedia The Partridge Family Jump to : navigation , search The Partridge Family Genre Musical sitcom Created by Bernard Slade Starring
Shirley Jones David Cassidy Danny Bonaduce Susan Dey Suzanne Crough Jeremy Gelbwaks Dave Madden Brian Forster Theme music composer Diane Hilderbrand Danny
Janssen Wes Farrell Opening theme `` When We 're Singin ' '' ( 1970 -- 71 ) `` C'mon , Get Happy '' ( 1971 -- 74 ) Composer ( s ) George Duning Benny Golson
Warren Barker Hugo Montenegro Shorty Rogers Country of origin United States Original language ( s ) English No. of seasons No. of episodes 96 ( list of episodes
) Production Executive producer ( s ) Bob Claver Producer ( s ) William Bickley Paul Junger Witt Dale McRaven Larry Rosen Mel Swope Cinematography Fred Jackman
, Jr .
</item 6>

<item 7>
The Partridge Family
The shows often contrast suburban life with the adventures of a show - business family on the road . After the first season , more of the show 's action takes
place in their hometown than on tour . History ( edit ) This section needs additional citations for verification . Please help improve this article by adding
citations to reliable sources . Unsourced material may be challenged and removed . ( May 2016 ) ( Learn how and when to remove this template message ) Created
for television by Bernard Slade , the series ' executive producer was Bob Claver . The show was inspired by and loosely based on The Cowsills , a family pop
music group that was famous in the late 1960s . In the show 's early development , the Cowsill children were considered by the producers , but because the
Cowsills were not trained actors and were too old for the roles as scripted , Slade and Claver abandoned that idea . Shirley Jones had already been signed as
mother Shirley Partridge and star of the show . The pilot was filmed in December 1969 . This unaired pilot differs from the pilot broadcast in 1970 .
</item 7>

<item 8>
*The Partridge Family* was an ABC musical sitcom starring Shirley Jones as a widowed mother and David Cassidy as her eldest child. Together with her five
children, she launched a music career. The single-camera series, produced by Screen Gems Television, ran for 25 minutes per episode from September 25, 1970, to
March 23, 1974, with monaural audio. Related shows included *Getting Together*, *Goober and the Ghost Chasers*, and *
</item 8>

<item 9>
*The Partridge Family* was an American musical sitcom created by Bernard Slade, with 96 episodes across an unspecified number of seasons. It starred Shirley
Jones, David Cassidy, Danny Bonaduce, Susan Dey, Suzanne Crough, Jeremy Gelbwaks, Dave Madden, and Brian Forster. Its opening themes were “When We’re Singin’”
(1970–71) and “C’mon, Get Happy” (1971–74). Bob Claver was executive
</item 9>

<item 10>
The Partridge Family
Actor David Cassidy sang lead on most of the records , and actor Shirley Jones can be heard on a few . But the other voices and all the instruments were
recorded by session musicians . The Partridge Family 's instrumental sound was provided by musicians informally known as The Wrecking Crew , who did many of the
pop hits of the day attributed to other performers . The actors pretended to play their instruments . None of the actors sang background ; they lip - synched to
recorded singing by The Love Generation , AKA The Ron Hicklin Singers , composed of brothers John and Tom Bahler , Jackie Ward ( who as Robin Ward had a # 14
hit `` Wonderful Summer '' on Dot Records in 1963 ) , and Ron Hicklin . Actor David Cassidy was originally to lip sync with the rest of the cast , but convinced
Farrell that he could sing and was allowed to join the studio ensemble as the lead singer . He and his step - mother , Shirley , are the only two show actors
allowed to sing for recording . The Partridge Family theme song , produced by Rogers , was shown over opening credits . More than one song was used as the theme
song .
</item 10>
=== END ITEM id=7ce67b12b2872c78 ===

=== ITEM id=832511c3994b2039 ===
Gold sentences and facts:
Sentence 0: No . overall No. in season Title Directed by Written by Original air date U.S. viewers ( millions ) 112 `` The Savior '' Eagle Egilsson Edward Kitsis & Adam Horowitz September 25 , 2016 ( 2016 - 09 - 25 ) 3.99 In the past , Aladdin , a `` Savior '' , is taunted by Jafar over the inevitable downfall of all Saviors . In Storybrooke , refugees arrive from the Land of Untold Stories . Emma experiences tremors and visions ; an arrested Hyde helps her receive an oracle 's vision of a hooded figure killing her . Guided by Morpheus , Gold enters Belle 's dreams to lift the sleeping curse ; he tries to renew their love , but she rejects him . `` Morpheus '' reveals he is really their unborn son , and desired this outcome . Belle wakes and leaves Gold . Henry convinces Regina that Hades may have lied about Robin Hood 's soul having been destroyed . Zelena moves in with Regina but , after learning her sister blames her for Robin Hood 's death , moves out and is approached by the Evil Queen . 113 `` A Bitter Draught '' Ron Underwood Andrew Chambliss & Dana Horgan October 2 , 2016 ( 2016 - 10 - 02 ) 3.72 In the past in the Enchanted Forest , the Evil Queen offers to aid the revenge plot of Edmond Dantes , The Count of Monte Cristo , if he kills Snow and David . When he falls in love with Snow 's handmaiden , he reneges and flees with her to the Land of Untold Stories to save her from poison . In Storybrooke , the Evil Queen recruits Zelena and uses Edmond 's heart to set him against Snow and David ; Regina kills Edmond to save the others , and realizes she retains the capacity for darkness . The Evil Queen declares that the undesirable `` untold stories '' will be played out . She gives David evidence that his father 's death was not accidental . Belle lives on Hook 's ship ; in a deal with Gold , the Evil Queen will not harm her or the baby . Emma tells Archie about her visions and notes that Regina is n't present in them and could be the hooded figure . 114 `` The Other Shoe '' Steve Pearlman Jane Espenson & Jerome Schwartz October 9 , 2016 ( 2016 - 10 - 09 ) 4.11 In the past in the Enchanted Forest , Cinderella meets Prince Thomas at the ball , but flees when she believes he loves her stepsister Clorinda ; Clorinda actually plans to elope with Thomas ' footman , Jacob , but her mother , Lady Tremaine , abducts her to the Land of Untold Stories after forcing Cinderella to betray Clorinda 's location . In Storybrooke , Ashley hopes to reconcile with Clorinda . Aided by the Evil Queen , Tremaine lures Ashley into a trap for her and Jacob . Ashley protects Jacob ; Clorinda reunites with him and forgives Cinderella , while Tremaine is arrested . Emma struggles to control her magic . Whale helps Jekyll recreate his serum , while the Evil Queen and Hyde form an alliance and she releases him . Snow White wants normality , but David secretly investigates his father 's death ; Gold provides information in exchange for David delivering a message to Belle . 115 `` Strange Case '' Alrick Riley David H. Goodman & Nelson Soler October 16 , 2016 ( 2016 - 10 - 16 ) 3.53 In the past in Victorian England , Jekyll petitions the Academy of Science for membership but Dr. Lydgate rejects him . Rumplestiltskin arrives and magically perfects Jekyll 's serum ; Jekyll transforms into Hyde , who blackmails Lydgate for Jekyll 's membership . Mary is Lydgate 's daughter and Jekyll 's unrequited love ; she and Hyde fall in love . Jekyll accidentally fatally defenestrates Mary ; he frames Hyde . In Storybrooke , Gold imprisons Belle on the ship to protect her from Hyde ; but it is Jekyll who tries to kill Belle in revenge for Rumplestiltskin 's role in Mary 's death . Hook saves Belle by killing Jekyll , and Hyde dies of identical wounds ; Regina asks Emma to kill her if necessary to stop the Evil Queen . Gold warns Belle that their child will need his protection . Mary Margaret resumes teaching , and her assistant Shirin inspires her to draw on her life as Snow . Shirin is secretly Princess Jasmine , working with the Oracle to search for Aladdin . 116 5 `` Street Rats '' Norman Buckley Edward Kitsis & Adam Horowitz October 23 , 2016 ( 2016 - 10 - 23 ) 3.40 In the past in Agrabah , Aladdin seeks a magical weapon for Jasmine , to save the kingdom from Jafar ; he is the weapon and a Savior . Jafar tempts Aladdin with magical shears that can sever him from his fate as the Savior and from his resulting death . Aladdin rejects Jafar and saves Jasmine . In Storybrooke , the Oracle is murdered . Emma helps Jasmine , hoping Aladdin 's survival ensures her own . Having faked his death , Aladdin reveals himself to Emma and confesses that he doomed Agrabah by using the shears , which he gives to her ; she asks Hook to dispose of them , but he secretly keeps them . The Evil Queen impersonates Archie to learn of Emma 's visions , and convinces Zelena to turn Archie back into a cricket . 117 6 `` Dark Waters '' Robert Duncan Andrew Chambliss & Brigitte Hales October 30 , 2016 ( 2016 - 10 - 30 ) 3.06 During the Dark Curse , Captain Nemo abducts Hook and invites him to join the crew `` family '' of his submarine , Nautilus , in their quest for the `` Mysterious Island '' , which is the Land of Untold Stories . Hook discovers the first mate is his half - brother , Liam , and decides to leave to avoid confrontation ; but Liam attacks him , instead mortally wounding Nemo , who steps in . In the present , Archie is freed . Henry confronts Hook over lying about the shears , and Liam imprisons them on Nautilus . Hook opens up and Henry accepts him as family ; they save each other and dispose of the shears . Modern medicine saves Nemo ; he , Liam , and Hook make amends . Emma and Aladdin discuss being a Savior . Jasmine reveals that Agrabah vanished . Snow supports Belle at her ultrasound . The Evil Queen seduces Gold and offers the retrieved shears in exchange for help taking Snow 's heart . 118 7 `` Heartless '' Ralph Hemecker Jane Espenson November 6 , 2016 ( 2016 - 11 - 06 ) 3.56 In the past , fugitive Snow and shepherd David separately travel to Longbourn ; she to flee the kingdom and he to sell his struggling farm . Without ever seeing each other 's faces , David saves Snow from a bounty hunter and she repays him with money to save the farm , deciding not to flee . Their hands touch and a magical sapling is created . In the present , the Evil Queen threatens to destroy Storybrooke with River of Lost Souls water unless Mary Margaret and David hand over their hearts . They learn the sapling could stop the Queen , who destroys it just after they find it . But it has shown them their previous meeting , and they conclude that their love can endure anything ; they surrender themselves . The Queen casts a sleeping curse on their hearts ; because they share Snow 's heart , the curse affects them both ; each can wake the other but immediately succumbs to the curse , so they can never be awake together . Regina uses Gold 's relationship with the Queen to separate the latter from Zelena , who earns Gold 's enmity by exposing his schemes to Belle . 119 8 `` I 'll Be Your Mirror '' Jennifer Lynch Jerome Schwartz & Leah Fong November 13 , 2016 ( 2016 - 11 - 13 ) 3.40 Regina and Emma try to magically trap the Evil Queen in the World Behind the Mirror , but she traps them there , where the Dragon is already imprisoned . Henry prepares for a date with Violet ; the Queen impersonates Regina and tries to manipulate him but he recognizes her . Emma and Regina discover Sidney was building a mirror portal out of the world and try and finish it , but the Evil Queen forces the Dragon to attack Emma and Regina , using his heart , and tries to make Henry kill the Dragon with the Hammer of Hephaestus ; he instead uses it to free his mothers . Mary Margaret and David adjust to living in shifts . Belle asks Zelena to help her escape Storybrooke via magical portal , and they form an alliance with Aladdin and Jasmine . Aladdin steals the Genie 's lamp and the Sorcerer 's wand from Gold 's shop , but Gold takes the wand back from Zelena and puts a magical tracking device on Belle . He asks the Queen to kill Zelena , as the connection between his and Zelena 's hearts protects her from him . 120 9 `` Changelings '' Mairzee Almas David H. Goodman & Brian Ridings November 27 , 2016 ( 2016 - 11 - 27 ) 3.28 In the Enchanted Forest , while served by Belle , Rumplestiltskin abducts a baby to use as bait for the Black Fairy , the mother who abandoned him ; she escapes without answering his questions . In the present , Belle receives advice from her unborn child while Gold becomes increasingly unhinged . Resenting Gold 's order to attack Zelena , the Evil Queen accelerates Belle 's pregnancy ; Mother Superior delivers Belle 's son , Gideon , and sends him into hiding ; Gold vows to find him . As the Genie has been freed , Aladdin takes his place . Zelena and Regina continue to feud . Emma obtains the sword from her vision . 121 10 `` Wish You Were Here '' Ron Underwood Edward Kitsis & Adam Horowitz December 4 , 2016 ( 2016 - 12 - 04 ) 3.27 The Black Fairy abducts Gideon . Emma discovers the sword can kill the Evil Queen without hurting Regina . The Queen steals the lamp and wishes Emma to an alternate reality where she was never the Savior ; Regina follows her . David and his allies take the lamp from the Queen , who is turned into a serpent by the hooded figure , revealed to be the adult Gideon . Jasmine wishes herself and Aladdin to Agrabah . In the `` wish realm '' , Emma lives as a princess until Regina traumatically restores her memory . Regina frees Rumplestiltskin in exchange for a magic bean , but the portal to Storybrooke closes while Regina is distracted by the appearance of Robin Hood . 122 11 `` Tougher Than the Rest '' Billy Gierhart Edward Kitsis & Adam Horowitz March 5 , 2017 ( 2017 - 03 - 05 ) 3.03 In 1990 Minnesota , a young Emma is told by a teenager that she has the power to change her fate . In present day , after they miss their way out of the Wish realm , Emma and Regina find another option for escape when they come across August , who agrees to help create a new portal . Regina goes to find Robin Hood to see if he would 've been better off without her . After Regina finds Robin , they 're caught by the Sheriff of Nottingham and later by the Wish realm Rumple . Robin believes Regina when she tells him about their being in an alternate reality . Emma discovers that August was the teenager she met back in Minnesota and inspires him to complete the portal . Emma , Regina , and Robin return to Storybrooke , where Gideon explains to Gold and Belle his intention to kill Emma so that he can acquire her savior powers and kill the Black Fairy , a move that has David furious and Belle concerned . Emma and Gideon finally meet for the battle , but Emma survives , choosing her own fate . Gideon retreats , proclaiming this far from over . Gold and Belle decide to work together for the good of their son . 123 12 `` Murder Most Foul '' Morgan Beggs Jerome Schwartz & Jane Espenson March 12 , 2017 ( 2017 - 03 - 12 ) 3.06 The events of how Robert , the father of David and James , was murdered are detailed in a flashback , in which Robert learns what happened to James after he ran away from King George . With help from Rumple , he tracks him to Pleasure Island , but their reunion is cut short by King George , who orders Robert 's execution . His death affects David in the present day after he sees visions of Robert and asks Hook to help him , despite his being reformed . When Hook becomes concerned that David is taking this investigation too far by going after Albert , he intervenes and brings David back to sanity . Hook is also worried about whether David will give him his blessing to marry Emma , but David does . Meanwhile , Regina tries to get the Wish Realm Robin assimilated into Storybrooke , but soon questions why he agreed to follow her , as it turns out that he had something else planned after he steals a chest from the vault . The truth as to who really murdered Robert is revealed when August gives Hook missing pages he took from Henry 's book . When Hook sees an illustration of David 's dad , he recognizes him and recalls murdering him during a robbery in the Enchanted Forest so that there was no witness to his crime . 124 13 `` Ill - Boding Patterns '' Ron Underwood Andrew Chambliss & Dana Horgan March 19 , 2017 ( 2017 - 03 - 19 ) 2.71 During the First Ogres War , Beowulf leads a fearsome charge on the battlefield in the Enchanted Forest of old . After every last soldier had been slain , Beowulf surrenders to death just as Rumplestiltskin intervenes , killing all the ogres . Hailed as a hero in the village , Beowulf is irked by Rumple 's new title . He creates a ruse to lead Rumple to his demise . Fearing he would not be able to follow through on protecting the villagers without Dark Magic , Rumple bestows his dagger upon Baelfire as an insurance policy . Following a turbulent exchange and threat to frame Rumplestiltskin as a murderer , Beowulf retreats . Feeling the pressure of his threat , Bae commands his father to kill him . Rumple later doses Bae with a memory potion to erase the ordeal from his mind and restore his innocence . In Storybrooke , Rumple recognizes a familiar pattern in Gideon , who makes strides in his quest to kill Emma Swan at the cost of The Blue Fairy 's vitality . Hook worries about the repercussions of telling Emma his newly unveiled secret . Meanwhile , Emma unknowingly forces a proposal after happening upon Hook 's engagement ring . Robin of Locksley attempts to flee the town , but to no avail . The Evil Queen escapes her reptilian confinement and indoctrinates Robin . 125 14 `` Page 23 '' Kate Woods David H. Goodman & Brigitte Hales March 26 , 2017 ( 2017 - 03 - 26 ) 2.85 In the Enchanted Forest , The Evil Queen is in search of a spell that will help her end Snow White for good , but her father has love in mind for his daughter and leads her on a quest to Cupid 's Arrow . However , The Evil Queen rebrands the arrow to follow hatred and discovers that the person she despises the most is herself . In Storybrooke , The Evil Queen and Regina face off in a final confrontation . Just when Regina wins the upper hand , she remembers the Evil Queen 's self - loating and evenly splits the light and the darkness in their hearts , to share their love and hate and allow the Evil Queen to reform . After Emma calls off the engagement , Hook decides to board the Nautilus with Captain Nemo . Hook later changes his mind following a conversation with Snow but , before he can return , Gideon hijacks the vessel and sends it out of Storybrooke . 126 15 `` A Wondrous Place '' Steve Pearlman Jane Espenson & Jerome Schwartz April 2 , 2017 ( 2017 - 04 - 02 ) 2.80 At a loss for hope , Regina suggests a `` girls ' night out '' for Emma with her and Snow . While at the bar , she confides in the mixologist and sheds a tear . Meanwhile , the Nautilus is transported to the Enchanted Forest , where Aladdin and Jasmine board . They have yet to locate Agrabah and are growing weary . After Hook and Jasmine confide in one another , the vessel is irrevocably damaged and Jasmine uses a genie wish to relocate the group to the island where Jafar is located . They discover a cabana belonging to Ariel and Prince Eric . Ariel has collected a genie lamp that houses Jafar . Jasmine is able to sway him into divulging the location of the hidden Agrabah before defeating him with magic dust . Through true love 's kiss , Aladdin and Jasmine , along with Hook and Ariel , are transported to the restored city and Aladdin 's genie curse is broken . Ariel supplies Hook with a means to reach Emma . Emma receives the message and Gideon , no longer masquerading as the mixologist , threatens that he will use her tears to keep Hook from returning to Storybrooke unless she helps him kill the Black Fairy . Flashbacks to Ariel and Jasmine 's first meeting show the princess falter under Jafar 's trickery . 127 16 `` Mother 's Little Helper '' Billy Gierhart Story by : Edward Kitsis & Adam Horowitz Teleplay by : Paul Karp April 9 , 2017 ( 2017 - 04 - 09 ) 2.60 In Storybrooke , Gold reveals to Emma and Snow that the Black Fairy is his mother and that the Dark Curse Regina cast was created by the Black Fairy . Emma agrees to help Gideon kill the Black Fairy , but is double - crossed by Gideon and left to die at the hands of a giant spider . Gideon opens a portal with the sword , allowing the Black Fairy to slip through into Storybrooke . Meanwhile in the Enchanted Forest , Hook wagers the Jolly Roger for a magic bean in a game of cards with Blackbeard . As Blackbeard and Hook go through the portal , it malfunctions thanks to Gideon and the duo ends up in Neverland , on the run from the remnants of Peter Pan 's Lost Boys . Back in Storybrooke , Henry experiences a disorienting shift in his Author powers , so he and Regina seek help from the only person who can aid them : the previous Author , Isaac . Isaac reveals to Regina and Henry that they are nearing the end of the story in which the savior must fight the Final Battle . Flashbacks reveal the abuses the Black Fairy inflicted on Gideon and that the Black Fairy ripped out Gideon 's heart and is controlling him to kill Emma . 128 17 `` Awake '' Sharat Raju Andrew Chambliss & Leah Fong April 16 , 2017 ( 2017 - 04 - 16 ) 2.51 Regina works feverishly to wake Snow and Charming from the Sleeping Curse . Gold comes face to face with the Black Fairy , the powerful enemy Emma must fight in the Final Battle . Hook , still in Neverland , meets up with Tiger Lily , who gives him a piece of an ancient fairy wand to aid Emma in defeating the Black Fairy . Gold confronts the Black Fairy over her manipulating Gideon with his heart and she maintains that Storybrooke would be destroyed if the two of them fought . A pixie flower allows Emma to open a doorway to Neverland and rescue Hook , who then re-proposes properly . The townsfolk gather around Emma 's sleeping parents and each take small amounts of the Sleeping Curse unto themselves , diluting it and allowing Snow and David to finally reunite . In 1993 , a pixie flower awakens Snow and Charming from their cursed selves , but an awakened Rumplestiltskin tells them they must make a difficult decision to ensure that Emma fulfills her destiny as the Savior . 129 18 `` Where Bluebirds Fly '' Michael Schultz David H. Goodman & Brigitte Hales April 23 , 2017 ( 2017 - 04 - 23 ) 2.69 In Zelena 's past , her childhood friend Stanum asks for her help in finding the Crimson Heart , a magical artifact that would be able to restore his human form after having been cursed by the Wicked Witch of the North to slowly turn into tin . Zelena 's refusal to give up her magic leads to her betraying Stanum , as she keeps the Heart and leaves him to become the Tin Man . In the Present , The Black Fairy tempts Zelena with an offer to join her , only to have Zelena oppose her instead . Regina is brought into the chaos by trying to stop her sister and they find the Black Fairy and Gideon in the crystal mines , with the Black Fairy tricking Zelena into utilizing her unstable Dark Magic to strengthen her for the Final Battle . Realizing that there is nothing left for her back in Oz , Zelena decides to help the residents fight back against the Black Fairy by using the Crimson Heart to give up her magic , thereby undoing the mess she made in the mines . Emma then uses her magic and one of the crystals to bring Mother Superior out of her coma with some help from Regina and Zelena . 130 19 `` The Black Fairy '' Alrick Riley Jerome Schwartz & Dana Horgan April 30 , 2017 ( 2017 - 04 - 30 ) 3.05 The origin of Rumplestiltskin 's true destiny is revealed , as his mother Fiona learns from the Blue Fairy and Tiger Lily that he was destined to become The Savior . When Fiona fears for her son and chooses power over protecting Rumple , it results in her becoming the Black Fairy , and the Blue Fairy banishing her to the Dark Realm . In the Present , as everyone looks for the missing wand , Gold brings Gideon and Emma into the Dream World to search for Gideon 's heart , where the events of the flashbacks unfold . Gold then confronts Fiona , but she reveals the real reason behind why he was abandoned . Gold later retrieves Gideon 's heart and tells everyone that the Black Fairy is dead , paving the way for Emma and Hook 's wedding . However , Gold and Fiona make a deal that will see the Final Battle go forward , with Emma being the Savior that will fight in it . Meanwhile , Regina helps Zelena adjust to life without her powers . 131 20 `` The Song in Your Heart '' Ron Underwood David H. Goodman & Andrew Chambliss May 7 , 2017 ( 2017 - 05 - 07 ) 2.87 In the past , Snow makes a wish that grants the entire Enchanted Forest the ability to use the gift of song as a strength , which also spreads to Oz as well . When Regina becomes affected by the wish , she finds a way to end it before the first curse takes place , but the Blue Fairy , who granted Snow the wish , tells Snow and David that the wish will be part of Emma 's destiny . In Emma 's past , her chance to make use of her talent is held back by being told she 'll be alone . This would come back to taunt Emma in the present day , as Fiona returns to use that weakness to take Emma 's heart before unleashing a new curse on her wedding day , with help from Gold by freezing her family . When Henry discovers the page in the book that reveals the song in Emma 's heart will make her stronger , not only does it embolden Emma , it stops Fiona from crushing her heart , but Fiona still tells Emma that the battle will continue . On her wedding day , Emma and Hook finally marry , just in time for the dark clouds of the Black Fairy 's curse to erupt from the clock tower and engulf them at 6 : 00 PM , setting up the final battle . 132 133 21 22 `` The Final Battle '' Steve Pearlman Ralph Hemecker Edward Kitsis & Adam Horowitz May 14 , 2017 ( 2017 - 05 - 14 ) 2.95 Fiona 's curse alters Storybrooke , institutionalizing Emma and leaving only Henry and Gold `` awake , '' while sending Emma 's other allies to their home realms . The Final Battle begins over Emma 's belief ; if she stops believing in the realms of story , they will be consumed by the Darkness and cease to exist . If all the realms are destroyed , Fiona would gain unchecked powers and be able to break the Laws of Magic at will . In the Enchanted Forest , the banished characters gather at Regina 's castle ; David and Hook obtain a magic bean and the Evil Queen sacrifices herself , holding off the Darkness to delay the group 's destruction . Fiona convinces Emma to destroy the book , but Henry hand - draws one that restores her belief in time to save everyone in the castle . Gold kills Fiona , who had hidden Belle and turned her into a coward ; this breaks the Curse , which restores everyone 's memories and reunites them in Storybrooke . However , Gideon is still under orders to kill Emma . Gold attempts to free Gideon 's heart ; however , a spell cast by the Black Fairy prevents him from doing so . Emma lays down her sword , allowing Gideon to impale her ; this releases Emma 's light magic and ends the battle . With both Gold and Emma having chosen good , Gideon is returned to infancy and the realms are restored , and those who died in the crisis are revived . The characters embark on their `` happy beginnings . '' Many years later in the New Enchanted Forest , an adult Henry vanishes while defending his daughter Lucy from an unidentified entity . Guided by Tiger Lily , Lucy travels to Seattle , where Henry no longer remembers her .
  Fact 0.0: Episode 112, 'The Savior', was directed by Eagle Egilsson and written by Edward Kitsis and Adam Horowitz.
  Fact 0.1: Episode 112 aired on September 25, 2016 and had 3.99 million U.S. viewers.
  Fact 0.2: In the past, Aladdin, a 'Savior', is taunted by Jafar over the inevitable downfall of all Saviors.
  Fact 0.3: In Storybrooke, refugees arrive from the Land of Untold Stories.
  Fact 0.4: Emma experiences tremors and visions.
  Fact 0.5: An arrested Hyde helps Emma receive an oracle's vision of a hooded figure killing her.
  Fact 0.6: Guided by Morpheus, Gold enters Belle's dreams to lift the sleeping curse.
  Fact 0.7: Gold tries to renew his love with Belle, but she rejects him.
  Fact 0.8: 'Morpheus' reveals he is really Belle and Gold's unborn son, and that he desired this outcome.
  Fact 0.9: Belle wakes and leaves Gold.
  Fact 0.10: Henry convinces Regina that Hades may have lied about Robin Hood's soul having been destroyed.
  Fact 0.11: Zelena moves in with Regina.
  Fact 0.12: After learning her sister Regina blames her for Robin Hood's death, Zelena moves out.
  Fact 0.13: Zelena is approached by the Evil Queen after moving out.
  Fact 0.14: Episode 113, 'A Bitter Draught', was directed by Ron Underwood and written by Andrew Chambliss and Dana Horgan.
  Fact 0.15: Episode 113 aired on October 2, 2016 and had 3.72 million U.S. viewers.
  Fact 0.16: In the past in the Enchanted Forest, the Evil Queen offers to aid the revenge plot of Edmond Dantes, The Count of Monte Cristo.
  Fact 0.17: The Evil Queen's offer to Edmond Dantes is conditional on him killing Snow and David.
  Fact 0.18: Edmond Dantes falls in love with Snow's handmaiden.
  Fact 0.19: Edmond Dantes reneges on the revenge plot and flees with the handmaiden to the Land of Untold Stories to save her from poison.
  Fact 0.20: In Storybrooke, the Evil Queen recruits Zelena.
  Fact 0.21: In Storybrooke, the Evil Queen uses Edmond's heart to set him against Snow and David.
  Fact 0.22: Regina kills Edmond to save the others.
  Fact 0.23: Regina realizes she retains the capacity for darkness.
  Fact 0.24: The Evil Queen declares that the undesirable 'untold stories' will be played out.
  Fact 0.25: The Evil Queen gives David evidence that his father's death was not accidental.
  Fact 0.26: Belle lives on Hook's ship.
  Fact 0.27: In a deal with Gold, the Evil Queen will not harm Belle or the baby.
  Fact 0.28: Emma tells Archie about her visions.
  Fact 0.29: Emma notes that Regina is not present in her visions and could be the hooded figure.
  Fact 0.30: Episode 114, 'The Other Shoe', was directed by Steve Pearlman and written by Jane Espenson and Jerome Schwartz.
  Fact 0.31: Episode 114 aired on October 9, 2016 and had 4.11 million U.S. viewers.
  Fact 0.32: In the past in the Enchanted Forest, Cinderella meets Prince Thomas at the ball.
  Fact 0.33: Cinderella flees when she believes Thomas loves her stepsister Clorinda.
  Fact 0.34: Clorinda actually plans to elope with Thomas' footman, Jacob.
  Fact 0.35: Clorinda's mother, Lady Tremaine, abducts Clorinda to the Land of Untold Stories.
  Fact 0.36: Lady Tremaine abducts Clorinda after forcing Cinderella to betray Clorinda's location.
  Fact 0.37: In Storybrooke, Ashley hopes to reconcile with Clorinda.
  Fact 0.38: Aided by the Evil Queen, Lady Tremaine lures Ashley into a trap for her and Jacob.
  Fact 0.39: Ashley protects Jacob.
  Fact 0.40: Clorinda reunites with Jacob and forgives Cinderella.
  Fact 0.41: Lady Tremaine is arrested.
  Fact 0.42: Emma struggles to control her magic.
  Fact 0.43: Whale helps Jekyll recreate his serum.
  Fact 0.44: The Evil Queen and Hyde form an alliance, and she releases him.
  Fact 0.45: Snow White wants normality.
  Fact 0.46: David secretly investigates his father's death.
  Fact 0.47: Gold provides information in exchange for David delivering a message to Belle.
  Fact 0.48: Episode 115, 'Strange Case', was directed by Alrick Riley and written by David H. Goodman and Nelson Soler.
  Fact 0.49: Episode 115 aired on October 16, 2016 and had 3.53 million U.S. viewers.
  Fact 0.50: In the past in Victorian England, Jekyll petitions the Academy of Science for membership.
  Fact 0.51: Dr. Lydgate rejects Jekyll's petition.
  Fact 0.52: Rumplestiltskin arrives and magically perfects Jekyll's serum.
  Fact 0.53: Jekyll transforms into Hyde.
  Fact 0.54: Hyde blackmails Lydgate for Jekyll's membership.
  Fact 0.55: Mary is Lydgate's daughter and Jekyll's unrequited love.
  Fact 0.56: Mary and Hyde fall in love.
  Fact 0.57: Jekyll accidentally fatally defenestrates Mary.
  Fact 0.58: Jekyll frames Hyde for Mary's death.
  Fact 0.59: In Storybrooke, Gold imprisons Belle on the ship to protect her from Hyde.
  Fact 0.60: Jekyll tries to kill Belle in revenge for Rumplestiltskin's role in Mary's death.
  Fact 0.61: Hook saves Belle by killing Jekyll.
  Fact 0.62: Hyde dies of identical wounds.
  Fact 0.63: Regina asks Emma to kill her if necessary to stop the Evil Queen.
  Fact 0.64: Gold warns Belle that their child will need his protection.
  Fact 0.65: Mary Margaret resumes teaching.
  Fact 0.66: Mary Margaret's assistant Shirin inspires her to draw on her life as Snow.
  Fact 0.67: Shirin is secretly Princess Jasmine.
  Fact 0.68: Princess Jasmine is working with the Oracle to search for Aladdin.
  Fact 0.69: Episode 116, 'Street Rats', is season episode 5 and was directed by Norman Buckley and written by Edward Kitsis and Adam Horowitz.
  Fact 0.70: Episode 116 aired on October 23, 2016 and had 3.40 million U.S. viewers.
  Fact 0.71: In the past in Agrabah, Aladdin seeks a magical weapon for Jasmine to save the kingdom from Jafar.
  Fact 0.72: Aladdin is the weapon and a Savior.
  Fact 0.73: Jafar tempts Aladdin with magical shears that can sever him from his fate as the Savior and from his resulting death.
  Fact 0.74: Aladdin rejects Jafar and saves Jasmine.
  Fact 0.75: In Storybrooke, the Oracle is murdered.
  Fact 0.76: Emma helps Jasmine, hoping Aladdin's survival ensures her own.
  Fact 0.77: Aladdin has faked his death.
  Fact 0.78: Aladdin reveals himself to Emma.
  Fact 0.79: Aladdin confesses that he doomed Agrabah by using the shears.
  Fact 0.80: Aladdin gives the shears to Emma.
  Fact 0.81: Emma asks Hook to dispose of the shears.
  Fact 0.82: Hook secretly keeps the shears.
  Fact 0.83: The Evil Queen impersonates Archie to learn of Emma's visions.
  Fact 0.84: The Evil Queen convinces Zelena to turn Archie back into a cricket.
  Fact 0.85: Episode 117, 'Dark Waters', is season episode 6 and was directed by Robert Duncan and written by Andrew Chambliss and Brigitte Hales.
  Fact 0.86: Episode 117 aired on October 30, 2016 and had 3.06 million U.S. viewers.
  Fact 0.87: During the Dark Curse, Captain Nemo abducts Hook.
  Fact 0.88: Captain Nemo invites Hook to join the crew 'family' of his submarine, Nautilus.
  Fact 0.89: The crew is on a quest for the 'Mysterious Island', which is the Land of Untold Stories.
  Fact 0.90: Hook discovers the first mate is his half-brother, Liam.
  Fact 0.91: Hook decides to leave to avoid confrontation.
  Fact 0.92: Liam attacks Hook, instead mortally wounding Nemo, who steps in.
  Fact 0.93: In the present, Archie is freed.
  Fact 0.94: Henry confronts Hook over lying about the shears.
  Fact 0.95: Liam imprisons Henry and Hook on Nautilus.
  Fact 0.96: Hook opens up and Henry accepts him as family.
  Fact 0.97: Henry and Hook save each other and dispose of the shears.
  Fact 0.98: Modern medicine saves Nemo.
  Fact 0.99: Nemo, Liam and Hook make amends.
  Fact 0.100: Emma and Aladdin discuss being a Savior.
  Fact 0.101: Jasmine reveals that Agrabah vanished.
  Fact 0.102: Snow supports Belle at her ultrasound.
  Fact 0.103: The Evil Queen seduces Gold.
  Fact 0.104: The Evil Queen offers Gold the retrieved shears in exchange for help taking Snow's heart.
  Fact 0.105: Episode 118, 'Heartless', is season episode 7 and was directed by Ralph Hemecker and written by Jane Espenson.
  Fact 0.106: Episode 118 aired on November 6, 2016 and had 3.56 million U.S. viewers.
  Fact 0.107: In the past, fugitive Snow and shepherd David separately travel to Longbourn.
  Fact 0.108: Snow travels to Longbourn to flee the kingdom.
  Fact 0.109: David travels to Longbourn to sell his struggling farm.
  Fact 0.110: Without seeing each other's faces, David saves Snow from a bounty hunter.
  Fact 0.111: Snow repays David with money to save the farm.
  Fact 0.112: Snow decides not to flee.
  Fact 0.113: Their hands touch and a magical sapling is created.
  Fact 0.114: In the present, the Evil Queen threatens to destroy Storybrooke with River of Lost Souls water.
  Fact 0.115: The Evil Queen's threat is unless Mary Margaret and David hand over their hearts.
  Fact 0.116: Mary Margaret and David learn the sapling could stop the Queen.
  Fact 0.117: The Queen destroys the sapling just after they find it.
  Fact 0.118: The sapling showed Mary Margaret and David their previous meeting.
  Fact 0.119: Mary Margaret and David conclude that their love can endure anything.
  Fact 0.120: Mary Margaret and David surrender themselves.
  Fact 0.121: The Queen casts a sleeping curse on their hearts.
  Fact 0.122: Because they share Snow's heart, the curse affects both of them.
  Fact 0.123: Each can wake the other but immediately succumbs to the curse.
  Fact 0.124: They can never be awake together.
  Fact 0.125: Regina uses Gold's relationship with the Queen to separate the Queen from Zelena.
  Fact 0.126: Zelena earns Gold's enmity by exposing his schemes to Belle.
  Fact 0.127: Episode 119, 'I'll Be Your Mirror', is season episode 8 and was directed by Jennifer Lynch and written by Jerome Schwartz and Leah Fong.
  Fact 0.128: Episode 119 aired on November 13, 2016 and had 3.40 million U.S. viewers.
  Fact 0.129: Regina and Emma try to magically trap the Evil Queen in the World Behind the Mirror.
  Fact 0.130: The Evil Queen traps Regina and Emma in the World Behind the Mirror.
  Fact 0.131: The Dragon is already imprisoned in the World Behind the Mirror.
  Fact 0.132: Henry prepares for a date with Violet.
  Fact 0.133: The Queen impersonates Regina and tries to manipulate Henry.
  Fact 0.134: Henry recognizes the Queen.
  Fact 0.135: Emma and Regina discover Sidney was building a mirror portal out of the world.
  Fact 0.136: Emma and Regina try to finish the mirror portal.
  Fact 0.137: The Evil Queen forces the Dragon to attack Emma and Regina, using his heart.
  Fact 0.138: The Evil Queen tries to make Henry kill the Dragon with the Hammer of Hephaestus.
  Fact 0.139: Henry instead uses the Hammer of Hephaestus to free his mothers.
  Fact 0.140: Mary Margaret and David adjust to living in shifts.
  Fact 0.141: Belle asks Zelena to help her escape Storybrooke via magical portal.
  Fact 0.142: Belle and Zelena form an alliance with Aladdin and Jasmine.
  Fact 0.143: Aladdin steals the Genie's lamp and the Sorcerer's wand from Gold's shop.
  Fact 0.144: Gold takes the wand back from Zelena.
  Fact 0.145: Gold puts a magical tracking device on Belle.
  Fact 0.146: Gold asks the Queen to kill Zelena.
  Fact 0.147: The connection between Gold's and Zelena's hearts protects Zelena from Gold.
  Fact 0.148: Episode 120, 'Changelings', is season episode 9 and was directed by Mairzee Almas and written by David H. Goodman and Brian Ridings.
  Fact 0.149: Episode 120 aired on November 27, 2016 and had 3.28 million U.S. viewers.
  Fact 0.150: In the Enchanted Forest, Rumplestiltskin, while served by Belle, abducts a baby.
  Fact 0.151: Rumplestiltskin uses the abducted baby as bait for the Black Fairy, the mother who abandoned him.
  Fact 0.152: The Black Fairy escapes without answering Rumplestiltskin's questions.
  Fact 0.153: In the present, Belle receives advice from her unborn child.
  Fact 0.154: In the present, Gold becomes increasingly unhinged.
  Fact 0.155: The Evil Queen resents Gold's order to attack Zelena.
  Fact 0.156: The Evil Queen accelerates Belle's pregnancy.
  Fact 0.157: Mother Superior delivers Belle's son, Gideon.
  Fact 0.158: Mother Superior sends Gideon into hiding.
  Fact 0.159: Gold vows to find Gideon.
  Fact 0.160: The Genie has been freed.
  Fact 0.161: Aladdin takes the Genie's place.
  Fact 0.162: Zelena and Regina continue to feud.
  Fact 0.163: Emma obtains the sword from her vision.
  Fact 0.164: Episode 121, 'Wish You Were Here', is season episode 10 and was directed by Ron Underwood and written by Edward Kitsis and Adam Horowitz.
  Fact 0.165: Episode 121 aired on December 4, 2016 and had 3.27 million U.S. viewers.
  Fact 0.166: The Black Fairy abducts Gideon.
  Fact 0.167: Emma discovers the sword can kill the Evil Queen without hurting Regina.
  Fact 0.168: The Queen steals the lamp.
  Fact 0.169: The Queen wishes Emma to an alternate reality where Emma was never the Savior.
  Fact 0.170: Regina follows the Queen into the alternate reality.
  Fact 0.171: David and his allies take the lamp from the Queen.
  Fact 0.172: The Queen is turned into a serpent by the hooded figure.
  Fact 0.173: The hooded figure is revealed to be the adult Gideon.
  Fact 0.174: Jasmine wishes herself and Aladdin to Agrabah.
  Fact 0.175: In the 'wish realm', Emma lives as a princess.
  Fact 0.176: Regina traumatically restores Emma's memory in the wish realm.
  Fact 0.177: Regina frees Rumplestiltskin in exchange for a magic bean.
  Fact 0.178: The portal to Storybrooke closes while Regina is distracted by the appearance of Robin Hood.
  Fact 0.179: Episode 122, 'Tougher Than the Rest', is season episode 11 and was directed by Billy Gierhart and written by Edward Kitsis and Adam Horowitz.
  Fact 0.180: Episode 122 aired on March 5, 2017 and had 3.03 million U.S. viewers.
  Fact 0.181: In 1990 Minnesota, a young Emma is told by a teenager that she has the power to change her fate.
  Fact 0.182: In the present day, Emma and Regina miss their way out of the Wish realm.
  Fact 0.183: Emma and Regina find another option for escape when they come across August.
  Fact 0.184: August agrees to help create a new portal.
  Fact 0.185: Regina goes to find Robin Hood to see if he would have been better off without her.
  Fact 0.186: Regina and Robin are caught by the Sheriff of Nottingham.
  Fact 0.187: Regina and Robin are later caught by the Wish realm Rumple.
  Fact 0.188: Robin believes Regina when she tells him about their being in an alternate reality.
  Fact 0.189: Emma discovers that August was the teenager she met in Minnesota.
  Fact 0.190: Emma inspires August to complete the portal.
  Fact 0.191: Emma, Regina and Robin return to Storybrooke.
  Fact 0.192: Gideon explains to Gold and Belle his intention to kill Emma.
  Fact 0.193: Gideon intends to kill Emma so that he can acquire her savior powers and kill the Black Fairy.
  Fact 0.194: Gideon's intention has David furious and Belle concerned.
  Fact 0.195: Emma and Gideon finally meet for the battle.
  Fact 0.196: Emma survives, choosing her own fate.
  Fact 0.197: Gideon retreats, proclaiming this far from over.
  Fact 0.198: Gold and Belle decide to work together for the good of their son.
  Fact 0.199: Episode 123, 'Murder Most Foul', is season episode 12 and was directed by Morgan Beggs and written by Jerome Schwartz and Jane Espenson.
  Fact 0.200: Episode 123 aired on March 12, 2017 and had 3.06 million U.S. viewers.
  Fact 0.201: A flashback details how Robert, the father of David and James, was murdered.
  Fact 0.202: In the flashback, Robert learns what happened to James after James ran away from King George.
  Fact 0.203: With help from Rumple, Robert tracks James to Pleasure Island.
  Fact 0.204: King George cuts short Robert and James' reunion.
  Fact 0.205: King George orders Robert's execution.
  Fact 0.206: Robert's death affects David in the present day.
  Fact 0.207: David sees visions of Robert and asks Hook to help him, despite Hook being reformed.
  Fact 0.208: Hook becomes concerned that David is taking the investigation too far by going after Albert.
  Fact 0.209: Hook intervenes and brings David back to sanity.
  Fact 0.210: Hook is worried about whether David will give him his blessing to marry Emma.
  Fact 0.211: David gives Hook his blessing to marry Emma.
  Fact 0.212: Regina tries to get the Wish Realm Robin assimilated into Storybrooke.
  Fact 0.213: Regina questions why Robin agreed to follow her.
  Fact 0.214: Robin had something else planned, and he steals a chest from the vault.
  Fact 0.215: The truth about who murdered Robert is revealed when August gives Hook missing pages he took from Henry's book.
  Fact 0.216: Hook sees an illustration of David's dad and recognizes him.
  Fact 0.217: Hook recalls murdering David's dad during a robbery in the Enchanted Forest so that there was no witness to his crime.
  Fact 0.218: Episode 124, 'Ill-Boding Patterns', is season episode 13 and was directed by Ron Underwood and written by Andrew Chambliss and Dana Horgan.
  Fact 0.219: Episode 124 aired on March 19, 2017 and had 2.71 million U.S. viewers.
  Fact 0.220: During the First Ogres War, Beowulf leads a fearsome charge on the battlefield in the Enchanted Forest of old.
  Fact 0.221: After every last soldier had been slain, Beowulf surrenders to death just as Rumplestiltskin intervenes.
  Fact 0.222: Rumplestiltskin kills all the ogres.
  Fact 0.223: Beowulf is irked by Rumple's new title as a hero in the village.
  Fact 0.224: Beowulf creates a ruse to lead Rumple to his demise.
  Fact 0.225: Fearing he would not be able to protect the villagers without Dark Magic, Rumple bestows his dagger upon Baelfire as an insurance policy.
  Fact 0.226: Beowulf retreats after a turbulent exchange and a threat to frame Rumplestiltskin as a murderer.
  Fact 0.227: Feeling the pressure of Beowulf's threat, Bae commands his father to kill him.
  Fact 0.228: Rumple later doses Bae with a memory potion to erase the ordeal from his mind and restore his innocence.
  Fact 0.229: In Storybrooke, Rumple recognizes a familiar pattern in Gideon.
  Fact 0.230: Gideon makes strides in his quest to kill Emma Swan at the cost of the Blue Fairy's vitality.
  Fact 0.231: Hook worries about the repercussions of telling Emma his newly unveiled secret.
  Fact 0.232: Emma unknowingly forces a proposal after happening upon Hook's engagement ring.
  Fact 0.233: Robin of Locksley attempts to flee the town, but to no avail.
  Fact 0.234: The Evil Queen escapes her reptilian confinement.
  Fact 0.235: The Evil Queen indoctrinates Robin.
  Fact 0.236: Episode 125, 'Page 23', is season episode 14 and was directed by Kate Woods and written by David H. Goodman and Brigitte Hales.
  Fact 0.237: Episode 125 aired on March 26, 2017 and had 2.85 million U.S. viewers.
  Fact 0.238: In the Enchanted Forest, the Evil Queen is in search of a spell that will help her end Snow White for good.
  Fact 0.239: The Evil Queen's father has love in mind for his daughter and leads her on a quest to Cupid's Arrow.
  Fact 0.240: The Evil Queen rebrands the arrow to follow hatred.
  Fact 0.241: The Evil Queen discovers that the person she despises the most is herself.
  Fact 0.242: In Storybrooke, the Evil Queen and Regina face off in a final confrontation.
  Fact 0.243: When Regina wins the upper hand, she remembers the Evil Queen's self-loathing.
  Fact 0.244: Regina evenly splits the light and the darkness in their hearts, to share their love and hate.
  Fact 0.245: Regina's split allows the Evil Queen to reform.
  Fact 0.246: After Emma calls off the engagement, Hook decides to board the Nautilus with Captain Nemo.
  Fact 0.247: Hook later changes his mind following a conversation with Snow.
  Fact 0.248: Before Hook can return, Gideon hijacks the Nautilus.
  Fact 0.249: Gideon sends the Nautilus out of Storybrooke.
  Fact 0.250: Episode 126, 'A Wondrous Place', is season episode 15 and was directed by Steve Pearlman and written by Jane Espenson and Jerome Schwartz.
  Fact 0.251: Episode 126 aired on April 2, 2017 and had 2.80 million U.S. viewers.
  Fact 0.252: At a loss for hope, Regina suggests a 'girls' night out' for Emma with her and Snow.
  Fact 0.253: At the bar, Regina confides in the mixologist and sheds a tear.
  Fact 0.254: In the Enchanted Forest, the Nautilus is transported.
  Fact 0.255: Aladdin and Jasmine board the Nautilus.
  Fact 0.256: Aladdin and Jasmine have yet to locate Agrabah and are growing weary.
  Fact 0.257: After Hook and Jasmine confide in one another, the vessel is irrevocably damaged.
  Fact 0.258: Jasmine uses a genie wish to relocate the group to the island where Jafar is located.
  Fact 0.259: The group discovers a cabana belonging to Ariel and Prince Eric.
  Fact 0.260: Ariel has collected a genie lamp that houses Jafar.
  Fact 0.261: Jasmine sways Jafar into divulging the location of the hidden Agrabah.
  Fact 0.262: Jasmine defeats Jafar with magic dust.
  Fact 0.263: Through true love's kiss, Aladdin and Jasmine, along with Hook and Ariel, are transported to the restored city.
  Fact 0.264: Aladdin's genie curse is broken.
  Fact 0.265: Ariel supplies Hook with a means to reach Emma.
  Fact 0.266: Emma receives the message.
  Fact 0.267: Gideon, no longer masquerading as the mixologist, threatens to use Emma's tears to keep Hook from returning to Storybrooke.
  Fact 0.268: Gideon's threat is unless Emma helps him kill the Black Fairy.
  Fact 0.269: Flashbacks to Ariel and Jasmine's first meeting show the princess faltering under Jafar's trickery.
  Fact 0.270: Episode 127, 'Mother's Little Helper', is season episode 16 and was directed by Billy Gierhart, with a story by Edward Kitsis and Adam Horowitz and a teleplay by Paul Karp.
  Fact 0.271: Episode 127 aired on April 9, 2017 and had 2.60 million U.S. viewers.
  Fact 0.272: In Storybrooke, Gold reveals to Emma and Snow that the Black Fairy is his mother.
  Fact 0.273: Gold reveals to Emma and Snow that the Dark Curse Regina cast was created by the Black Fairy.
  Fact 0.274: Emma agrees to help Gideon kill the Black Fairy.
  Fact 0.275: Gideon double-crosses Emma.
  Fact 0.276: Gideon leaves Emma to die at the hands of a giant spider.
  Fact 0.277: Gideon opens a portal with the sword.
  Fact 0.278: The portal allows the Black Fairy to slip through into Storybrooke.
  Fact 0.279: In the Enchanted Forest, Hook wagers the Jolly Roger for a magic bean in a game of cards with Blackbeard.
  Fact 0.280: As Blackbeard and Hook go through the portal, it malfunctions thanks to Gideon.
  Fact 0.281: Blackbeard and Hook end up in Neverland.
  Fact 0.282: Blackbeard and Hook are on the run from the remnants of Peter Pan's Lost Boys.
  Fact 0.283: In Storybrooke, Henry experiences a disorienting shift in his Author powers.
  Fact 0.284: Henry and Regina seek help from the previous Author, Isaac.
  Fact 0.285: Isaac reveals to Regina and Henry that they are nearing the end of the story.
  Fact 0.286: In the story, the savior must fight the Final Battle.
  Fact 0.287: Flashbacks reveal the abuses the Black Fairy inflicted on Gideon.
  Fact 0.288: The Black Fairy ripped out Gideon's heart.
  Fact 0.289: The Black Fairy is controlling Gideon to kill Emma.
  Fact 0.290: Episode 128, 'Awake', is season episode 17 and was directed by Sharat Raju and written by Andrew Chambliss and Leah Fong.
  Fact 0.291: Episode 128 aired on April 16, 2017 and had 2.51 million U.S. viewers.
  Fact 0.292: Regina works feverishly to wake Snow and Charming from the Sleeping Curse.
  Fact 0.293: Gold comes face to face with the Black Fairy.
  Fact 0.294: The Black Fairy is the powerful enemy Emma must fight in the Final Battle.
  Fact 0.295: In Neverland, Hook meets up with Tiger Lily.
  Fact 0.296: Tiger Lily gives Hook a piece of an ancient fairy wand to aid Emma in defeating the Black Fairy.
  Fact 0.297: Gold confronts the Black Fairy over her manipulating Gideon with his heart.
  Fact 0.298: The Black Fairy maintains that Storybrooke would be destroyed if she and Gold fought.
  Fact 0.299: A pixie flower allows Emma to open a doorway to Neverland.
  Fact 0.300: Emma rescues Hook.
  Fact 0.301: Hook re-proposes properly.
  Fact 0.302: The townsfolk gather around Emma's sleeping parents.
  Fact 0.303: The townsfolk each take small amounts of the Sleeping Curse unto themselves, diluting it.
  Fact 0.304: Diluting the curse allows Snow and David to finally reunite.
  Fact 0.305: In 1993, a pixie flower awakens Snow and Charming from their cursed selves.
  Fact 0.306: An awakened Rumplestiltskin tells Snow and Charming they must make a difficult decision to ensure that Emma fulfills her destiny as the Savior.
  Fact 0.307: Episode 129, 'Where Bluebirds Fly', is season episode 18 and was directed by Michael Schultz and written by David H. Goodman and Brigitte Hales.
  Fact 0.308: Episode 129 aired on April 23, 2017 and had 2.69 million U.S. viewers.
  Fact 0.309: In Zelena's past, her childhood friend Stanum asks for her help in finding the Crimson Heart.
  Fact 0.310: The Crimson Heart is a magical artifact that would restore Stanum's human form.
  Fact 0.311: Stanum was cursed by the Wicked Witch of the North to slowly turn into tin.
  Fact 0.312: Zelena refuses to give up her magic.
  Fact 0.313: Zelena betrays Stanum by keeping the Heart.
  Fact 0.314: Zelena leaves Stanum to become the Tin Man.
  Fact 0.315: In the present, the Black Fairy tempts Zelena with an offer to join her.
  Fact 0.316: Zelena opposes the Black Fairy instead.
  Fact 0.317: Regina is brought into the chaos by trying to stop her sister.
  Fact 0.318: Regina and Zelena find the Black Fairy and Gideon in the crystal mines.
  Fact 0.319: The Black Fairy tricks Zelena into utilizing her unstable Dark Magic to strengthen her for the Final Battle.
  Fact 0.320: Zelena realizes there is nothing left for her back in Oz.
  Fact 0.321: Zelena decides to help the residents fight back against the Black Fairy.
  Fact 0.322: Zelena uses the Crimson Heart to give up her magic.
  Fact 0.323: Giving up her magic undoes the mess Zelena made in the mines.
  Fact 0.324: Emma uses her magic and one of the crystals to bring Mother Superior out of her coma.
  Fact 0.325: Regina and Zelena help Emma bring Mother Superior out of her coma.
  Fact 0.326: Episode 130, 'The Black Fairy', is season episode 19 and was directed by Alrick Riley and written by Jerome Schwartz and Dana Horgan.
  Fact 0.327: Episode 130 aired on April 30, 2017 and had 3.05 million U.S. viewers.
  Fact 0.328: The origin of Rumplestiltskin's true destiny is revealed.
  Fact 0.329: Rumplestiltskin's mother, Fiona, learns from the Blue Fairy and Tiger Lily that he was destined to become The Savior.
  Fact 0.330: Fiona fears for her son and chooses power over protecting Rumple.
  Fact 0.331: Fiona becomes the Black Fairy.
  Fact 0.332: The Blue Fairy banishes Fiona to the Dark Realm.
  Fact 0.333: In the present, everyone looks for the missing wand.
  Fact 0.334: Gold brings Gideon and Emma into the Dream World to search for Gideon's heart.
  Fact 0.335: The events of the flashbacks unfold in the Dream World.
  Fact 0.336: Gold confronts Fiona.
  Fact 0.337: Fiona reveals the real reason behind why Gold was abandoned.
  Fact 0.338: Gold retrieves Gideon's heart.
  Fact 0.339: Gold tells everyone that the Black Fairy is dead.
  Fact 0.340: The Black Fairy's death paves the way for Emma and Hook's wedding.
  Fact 0.341: Gold and Fiona make a deal that will see the Final Battle go forward.
  Fact 0.342: Emma will be the Savior that fights in the Final Battle.
  Fact 0.343: Regina helps Zelena adjust to life without her powers.
  Fact 0.344: Episode 131, 'The Song in Your Heart', is season episode 20 and was directed by Ron Underwood and written by David H. Goodman and Andrew Chambliss.
  Fact 0.345: Episode 131 aired on May 7, 2017 and had 2.87 million U.S. viewers.
  Fact 0.346: In the past, Snow makes a wish that grants the entire Enchanted Forest the ability to use the gift of song as a strength.
  Fact 0.347: The wish also spreads to Oz.
  Fact 0.348: Regina becomes affected by the wish.
  Fact 0.349: Regina finds a way to end the wish before the first curse takes place.
  Fact 0.350: The Blue Fairy, who granted Snow the wish, tells Snow and David that the wish will be part of Emma's destiny.
  Fact 0.351: In Emma's past, her chance to make use of her talent is held back by being told she'll be alone.
  Fact 0.352: The weakness of being told she'll be alone comes back to taunt Emma in the present day.
  Fact 0.353: Fiona returns to use that weakness to take Emma's heart.
  Fact 0.354: Fiona unleashes a new curse on Emma's wedding day, with help from Gold.
  Fact 0.355: The curse freezes Emma's family.
  Fact 0.356: Henry discovers the page in the book that reveals the song in Emma's heart will make her stronger.
  Fact 0.357: The song embolden Emma.
  Fact 0.358: The song stops Fiona from crushing Emma's heart.
  Fact 0.359: Fiona tells Emma that the battle will continue.
  Fact 0.360: On her wedding day, Emma and Hook finally marry.
  Fact 0.361: Dark clouds of the Black Fairy's curse erupt from the clock tower at 6:00 PM.
  Fact 0.362: The dark clouds engulf Emma and Hook.
  Fact 0.363: The curse sets up the final battle.
  Fact 0.364: Season episodes 21 and 22, 'The Final Battle', were directed by Steve Pearlman and Ralph Hemecker and written by Edward Kitsis and Adam Horowitz.
  Fact 0.365: The Final Battle aired on May 14, 2017 and had 2.95 million U.S. viewers.
  Fact 0.366: Fiona's curse alters Storybrooke.
  Fact 0.367: Fiona's curse institutionalizes Emma.
  Fact 0.368: Fiona's curse leaves only Henry and Gold 'awake'.
  Fact 0.369: Fiona's curse sends Emma's other allies to their home realms.
  Fact 0.370: The Final Battle begins over Emma's belief.
  Fact 0.371: If Emma stops believing in the realms of story, they will be consumed by the Darkness and cease to exist.
  Fact 0.372: If all the realms are destroyed, Fiona would gain unchecked powers.
  Fact 0.373: If all the realms are destroyed, Fiona would be able to break the Laws of Magic at will.
  Fact 0.374: In the Enchanted Forest, the banished characters gather at Regina's castle.
  Fact 0.375: David and Hook obtain a magic bean.
  Fact 0.376: The Evil Queen sacrifices herself.
  Fact 0.377: The Evil Queen holds off the Darkness to delay the group's destruction.
  Fact 0.378: Fiona convinces Emma to destroy the book.
  Fact 0.379: Henry hand-draws a new book.
  Fact 0.380: Henry's hand-drawn book restores Emma's belief in time to save everyone in the castle.
  Fact 0.381: Gold kills Fiona.
  Fact 0.382: Fiona had hidden Belle and turned her into a coward.
  Fact 0.383: Gold's killing of Fiona breaks the Curse.
  Fact 0.384: Breaking the Curse restores everyone's memories.
  Fact 0.385: Breaking the Curse reunites everyone in Storybrooke.
  Fact 0.386: Gideon is still under orders to kill Emma.
  Fact 0.387: Gold attempts to free Gideon's heart.
  Fact 0.388: A spell cast by the Black Fairy prevents Gold from freeing Gideon's heart.
  Fact 0.389: Emma lays down her sword, allowing Gideon to impale her.
  Fact 0.390: Gideon impaling Emma releases Emma's light magic and ends the battle.
  Fact 0.391: Gold and Emma have both chosen good.
  Fact 0.392: Gideon is returned to infancy.
  Fact 0.393: The realms are restored.
  Fact 0.394: Those who died in the crisis are revived.
  Fact 0.395: The characters embark on their 'happy beginnings'.
  Fact 0.396: Many years later in the New Enchanted Forest, an adult Henry vanishes while defending his daughter Lucy from an unidentified entity.
  Fact 0.397: Guided by Tiger Lily, Lucy travels to Seattle.
  Fact 0.398: In Seattle, Henry no longer remembers Lucy.

Retrieved items:
<item 1>
Once Upon a Time (season 6)
Cast and characters ( edit ) Main article : List of Once Upon a Time characters Regular ( edit ) Ginnifer Goodwin as Snow White / Mary Margaret Blanchard
Jennifer Morrison as Emma Swan Lana Parrilla as Evil Queen / Regina Mills Josh Dallas as Prince Charming / David Nolan Emilie de Ravin as Belle / Lacey Colin
O'Donoghue as Captain Hook / Killian Jones Jared S. Gilmore as Henry Mills Rebecca Mader as Wicked Witch of the West / Zelena Robert Carlyle as Rumplestiltskin
/ Mr. Gold Recurring ( edit ) Beverley Elliott as Widow Lucas / Granny Raphael Sbarge as Jiminy Cricket / Dr. Archie Hopper Lee Arenberg as Dreamy / Grumpy /
Leroy David Paul Grove as Doc Gabe Khouth as Sneezy / Mr. Clark Faustino Di Bauda as Sleepy / Walter Mig Macario as Bashful Geoff Gustafson as Stealthy Tony
Amendola as Geppetto / Marco Keegan Connor Tracy as the Blue Fairy / Mother Superior Eion Bailey as Pinocchio / August Booth David Anders as Victor Frankenstein
/ Dr.
</item 1>

<item 2>
Once Upon a Time (season 6)
In April 2017 , Horowitz and Kitsis confirmed that a group of characters will indeed have their stories wrapped up by the end of the season in acknowledging the
potential cast changes , saying : `` We planned this finale from the beginning of the year , so whoever stays and whoever goes ... all those questions have
already been dealt with . The audience does not have to fear ( anything feeling ) incomplete . '' On May 8 , Morrison confirmed that she had declined an offer
to remain on the show and that the sixth season would be her last , signaling the end of Emma 's time on the show as the main protagonist . On May 11 , Rebecca
Mader announced that she would also be leaving the show at the end of the season , citing creative decisions beyond her control . On May 12 , season six was
announced to be the last for four additional main actors : Ginnifer Goodwin , Josh Dallas , Jared Gilmore , and Emilie de Ravin . Thus , characters seeing their
storylines wrapped up in the season six finale include Emma Swan , Snow White , Prince Charming , Belle , and Zelena .
</item 2>

<item 3>
Once Upon a Time (season 6)
Emma and Gideon finally meet for the battle , but Emma survives , choosing her own fate . Gideon retreats , proclaiming this far from over . Gold and Belle
decide to work together for the good of their son . 123 12 `` Murder Most Foul '' Morgan Beggs Jerome Schwartz & Jane Espenson March 12 , 2017 ( 2017 - 03 - 12
) 3.06 The events of how Robert , the father of David and James , was murdered are detailed in a flashback , in which Robert learns what happened to James after
he ran away from King George . With help from Rumple , he tracks him to Pleasure Island , but their reunion is cut short by King George , who orders Robert 's
execution . His death affects David in the present day after he sees visions of Robert and asks Hook to help him , despite his being reformed . When Hook
becomes concerned that David is taking this investigation too far by going after Albert , he intervenes and brings David back to sanity . Hook is also worried
about whether David will give him his blessing to marry Emma , but David does .
</item 3>

<item 4>
Once Upon a Time (season 6)
Gold kills Fiona , who had hidden Belle and turned her into a coward ; this breaks the Curse , which restores everyone 's memories and reunites them in
Storybrooke . However , Gideon is still under orders to kill Emma . Gold attempts to free Gideon 's heart ; however , a spell cast by the Black Fairy prevents
him from doing so . Emma lays down her sword , allowing Gideon to impale her ; this releases Emma 's light magic and ends the battle . With both Gold and Emma
having chosen good , Gideon is returned to infancy and the realms are restored , and those who died in the crisis are revived . The characters embark on their
`` happy beginnings . '' Many years later in the New Enchanted Forest , an adult Henry vanishes while defending his daughter Lucy from an unidentified entity .
Guided by Tiger Lily , Lucy travels to Seattle , where Henry no longer remembers her .
</item 4>

<item 5>
Once Upon a Time (season 6)
132 133 21 22 `` The Final Battle '' Steve Pearlman Ralph Hemecker Edward Kitsis & Adam Horowitz May 14 , 2017 ( 2017 - 05 - 14 ) 2.95 Fiona 's curse alters
Storybrooke , institutionalizing Emma and leaving only Henry and Gold `` awake , '' while sending Emma 's other allies to their home realms . The Final Battle
begins over Emma 's belief ; if she stops believing in the realms of story , they will be consumed by the Darkness and cease to exist . If all the realms are
destroyed , Fiona would gain unchecked powers and be able to break the Laws of Magic at will . In the Enchanted Forest , the banished characters gather at
Regina 's castle ; David and Hook obtain a magic bean and the Evil Queen sacrifices herself , holding off the Darkness to delay the group 's destruction . Fiona
convinces Emma to destroy the book , but Henry hand - draws one that restores her belief in time to save everyone in the castle .
</item 5>

<item 6>
Once Upon a Time (season 6)
Contents ( hide ) 1 Premise 2 Cast and characters 2.1 Regular 2.2 Recurring 2.3 Guest 3 Episodes 4 Production 4.1 Development 4.1. 1 Musical episode 4.2 Casting
5 Ratings 6 References 7 External links Premise ( edit ) The residents of Storybrooke are faced with threats from the Evil Queen and Mr. Hyde , both with
different agendas following the arrival of refugees from the Land of Untold Stories . This leads to events to test Emma 's savior abilities that results in the
arrival of the Black Fairy to begin the Final Battle that was prophesied centuries ago . As dark and light collides , a new , and darker curse is unleashed ,
leading to the fall of all the realms with magic . The Final Battle is fought and won , leading to the heroes earning their happy beginnings , but a new
adventure begins in the far future for an adult Henry involving his daughter Lucy , and another curse looming across the New Enchanted Forest inhabitants .
</item 6>

<item 7>
Once Upon a Time (season 6)
Guided by Tiger Lily , Lucy travels to Seattle , where Henry no longer remembers her . Production ( edit ) Development ( edit ) Executive producers Adam
Horowitz and Edward Kitsis announced that they were ending the half - season arc structure that was seen in seasons three through five , with Horowitz saying ,
`` We 're also planning a 22 - episode story as opposed to breaking it up into two halves this year . It has been really exciting and fun . '' Kitsis added : ``
We are changing around what we 're doing this year and going back to that season 1 mentality of small town stories and smaller arcs . '' This season will focus
on Storybrooke as the main setting , but will also show new realms and there will be the exploration of the Savior 's mythology . Kitsis and Horowitz hired two
directors from their Freeform series Dead of Summer , Norman Buckley , Mairzee Almas , and Michael Schultz helm episodes of the season . Jennifer Lynch directed
the eighth episode of the season , `` I 'll Be Your Mirror '' .
</item 7>

<item 8>
Once Upon a Time (season 6)
^ Jump up to : Abrams , Natalie ( May 11 , 2017 ) . `` Once Upon a Time : Rebecca Mader exiting after season 6 '' . Entertainment Weekly . Retrieved May 11 ,
2017 . ^ Jump up to : Abrams , Natalie ( May 12 , 2017 ) . `` Once Upon a Time : Ginnifer Goodwin , Josh Dallas , Emilie de Ravin and Jared Gilmore exit '' .
Entertainment Weekly . Retrieved May 12 , 2017 . ^ Jump up to : Adam Horowitz ( @ AdamHorowitzLA ) ( July 6 , 2016 ) . `` @ IdkiloveMinaj if they 're not , not
sure what they 're doing in Vancouver '' ( Tweet ) -- via Twitter . ^ Jump up to : Adam Horowitz ( @ AdamHorowitzLA ) ( May 29 , 2016 ) . `` @ Pirate_Mia I hope
you enjoy what we have in store for Hook and Emma next season ! '' ( Tweet ) -- via Twitter .
</item 8>

<item 9>
Once Upon a Time (season 6)
Once Upon a Time ( season 6 ) - wikipedia Once Upon a Time ( season 6 ) Jump to : navigation , search Once Upon a Time ( season 6 ) DVD cover Starring Ginnifer
Goodwin Jennifer Morrison Lana Parrilla Josh Dallas Emilie de Ravin Colin O'Donoghue Jared S. Gilmore Rebecca Mader Robert Carlyle Country of origin United
States No. of episodes 22 Release Original network ABC Original release September 25 , 2016 ( 2016 - 09 - 25 ) -- May 14 , 2017 ( 2017 - 05 - 14 ) Season
chronology ← Previous Season 5 Next → Season 7 List of Once Upon a Time episodes The sixth season of the American ABC fantasy - drama Once Upon a Time was
ordered on March 3 , 2016 . It debuted on September 25 , 2016 , and concluded on May 14 , 2017 . In January 2017 , it was stated that the sixth season would end
the main storyline , and for a seventh season , the series would be softly rebooted with a new storyline .
</item 9>

<item 10>
Once Upon a Time (season 6)
Existing fictional characters introduced to the series during the season include Aladdin , Princess Jasmine , the Count of Monte Cristo , Captain Nemo , Lady
Tremaine , Beowulf , Tiger Lily and the Tin Man . Original new characters include Gideon , the Black Fairy , Mary Lydgate , and Robert . The show also
reintroduced Jafar and Dr. Arthur Lydgate , who previously appeared in Once Upon a Time in Wonderland . This season also marks the final appearance of Emma Swan
( Jennifer Morrison ) as a series regular . Morrison announced she would be departing the series after the sixth - season finale , but if the series receives a
seventh season renewal she has agreed to appear in at least one episode . After serving as a series regular for two seasons , Rebecca Mader also announced that
season six would be her last on the show as a regular . Shortly after , Adam Horowitz and Edward Kitsis announced that original cast members Ginnifer Goodwin ,
Josh Dallas , and Jared Gilmore , as well as Emilie de Ravin who joined the main cast in season two , would also exit the show at the end of the season .
</item 10>
=== END ITEM id=832511c3994b2039 ===

=== ITEM id=f4c684a05f8c592c ===
Gold sentences and facts:
Sentence 0: No . overall No. in season Title Directed by Written by Original release date `` Chapter One : The Vanishing of Will Byers '' The Duffer Brothers The Duffer Brothers July 15 , 2016 ( 2016 - 07 - 15 ) On November 6 , 1983 , in a U.S. Department of Energy laboratory in the town of Hawkins , Indiana , a scientist is attacked by an unseen creature . While bicycling home from a Dungeons & Dragons session with his friends , 12 - year - old Will Byers encounters the creature and vanishes . The next day , a buzz - cut young girl wearing a hospital gown steals food from a local diner . The owner , Benny , takes pity on her and feeds her before calling social services . From a tattoo on her arm , he learns that her name is Eleven . A woman posing as a social worker arrives and murders Benny . Armed men search the diner for Eleven , but she escapes . Will 's mother Joyce believes she hears Will 's voice on a distorted phone call , but her phone short circuits . Will 's friends Mike , Dustin , and Lucas search for Will in the woods and find Eleven . `` Chapter Two : The Weirdo on Maple Street '' The Duffer Brothers The Duffer Brothers July 15 , 2016 ( 2016 - 07 - 15 ) The boys bring Eleven to Mike 's house , where she sleeps in the basement . The next day , Mike nicknames her `` El '' . She says that `` bad people '' are looking for her and refuses to meet Mike 's parents . Scientists from the laboratory find a substance oozing from the walls of Joyce 's home . At Mike 's home , Eleven recognizes and points out Will in a photo . Dustin and Lucas want to inform Mike 's parents about Eleven , but she uses psychokinesis to stop them . While searching for Will , Mr. Clarke , the boys ' science teacher , discovers a scrap of a hospital gown outside the laboratory grounds . Nancy goes with her friend Barb to a party at her boyfriend Steve 's house . Will 's brother Jonathan investigates the woods where Will went missing . Hearing screaming , he runs to help but finds only Steve , Nancy , and their friends roughhousing around Steve 's swimming pool . He secretly photographs them . Barb , left alone by the poolside , vanishes . Joyce receives another call from Will , hears music from his room , and sees something coming through the wall . `` Chapter Three : Holly , Jolly '' Shawn Levy Jessica Mecklenburg July 15 , 2016 ( 2016 - 07 - 15 ) Barb wakes up in an empty and decaying pool . She screams for Nancy and attempts to climb out but is dragged down by something unseen . Joyce strings Christmas lights around her home to talk to Will , who can turn them on and off . Hawkins Lab director Dr. Martin Brenner allows Hopper to view the lab 's security camera footage . Hopper realizes it is fake . He and his deputies research Hawkins Lab , Brenner , and a woman named Terry Ives who claimed her daughter was taken by Brenner . Eleven has a flashback in which Brenner , whom she calls `` Papa , '' has her put in solitary confinement for refusing to telekinetically harm a cat . Nancy worries about Barb , who is missing . Steve and his friends discover Jonathan 's photographs . They destroy them and his camera . Nancy recovers a photo of Barb by the poolside and returns to Steve 's house to search for her . Nancy finds Barb 's car and sees the creature in the woods . Joyce establishes a code with Will using the lights , which he uses to tell her that he is alive but unsafe . He tells Joyce to run as a creature begins to climb through her wall . Will 's apparent body is discovered in the water at a quarry . `` Chapter Four : The Body '' Shawn Levy Justin Doble July 15 , 2016 ( 2016 - 07 - 15 ) Eleven proves to the boys that Will is still alive by making contact through Mike 's walkie - talkie . Joyce goes to the morgue to view the body and refuses to believe it is Will . The boys give Eleven a makeover so they can sneak her into school . Joyce hears Will in her living room wall . Tearing away the wallpaper , she sees him behind a membrane . At school , Eleven uses Mr. Clarke 's ham radio to channel Will talking to his mother . Joyce demolishes the wall with an axe but reveals only her front porch . Nancy discovers a figure behind Barb in Jonathan 's photo . Jonathan realizes that the figure matches his mother 's description of the creature : a humanoid figure with long arms and no face . Nancy tells the police about her missing friend and subsequently admits to her mother that she slept with Steve . When she tells Steve about the police , she is irritated that he cares only about not getting in trouble with his father . Hopper confronts the state trooper who found Will 's body and beats him until he admits he was ordered to lie . Hopper goes to the morgue and finds that Will 's supposed body is a dummy . He breaks into Hawkins laboratory . 5 5 `` Chapter Five : The Flea and the Acrobat '' The Duffer Brothers Alison Tatlock July 15 , 2016 ( 2016 - 07 - 15 ) Will 's father Lonnie assures Joyce her experiences are hallucinations . Hopper discovers a portal in the basement of the lab but is knocked out by the lab 's guards . He wakes at home and discovers a hidden microphone . Sheriff 's deputies inform Hopper that Barb 's car was found , but Hopper finds the state 's involvement suspicious . The boys determine Will is trapped in an alternate dimension Eleven calls the `` Upside Down , '' and he is being pursued by a monster similar to a Demogorgon . After Will 's funeral , science teacher Mr. Clarke tells the boys that a high - energy spacetime tear could create a passage between dimensions . Using their compasses , they search for a disruption in the Earth 's electromagnetic field . Eleven remembers encountering the creature during a lab experiment , and afraid , she redirects the compasses . Lucas notices and confronts her . Mike defends her , he and Lucas fight , and Eleven telekinetically flings Lucas off Mike . While Dustin and Mike tend to the unconscious Lucas , Eleven disappears . In the woods , Nancy and Jonathan find a wounded deer . The creature drags the deer away , and Nancy follows the trail of blood to a portal to the Upside Down . She crawls through and discovers the creature feasting upon the deer . She snaps a twig and draws the creature 's attention . 6 6 `` Chapter Six : The Monster '' The Duffer Brothers Jessie Nickson - Lopez July 15 , 2016 ( 2016 - 07 - 15 ) Jonathan pulls Nancy back through the portal and saves her from the monster . Later , Steve sees them together through her bedroom window and assumes they are dating . Nancy and Jonathan resolve to kill the monster and purchase supplies from an army surplus store . During a fist fight with Steve , Jonathan inadvertently assaults a police officer and is arrested . Joyce and Hopper track down Terry Ives , who is catatonic and tended by her sister Becky . Becky explains that Terry underwent Project MKUltra testing while unknowingly pregnant ; she believed her daughter Jane was kidnapped by Brenner at birth . Joyce and Hopper conclude Eleven is likely Jane . Eleven shoplifts Eggo waffles . She recalls being asked to contact the monster and accidentally opening the portal to the Upside Down in the laboratory basement . While searching for Eleven , Mike and Dustin are ambushed by school bullies Troy and James . Troy holds Dustin at knifepoint and demands Mike make a suicidal jump into the lake where Will 's body was discovered . Mike jumps but is levitated to safety by Eleven , who scares away the bullies . Lucas realizes agents from the laboratory are heading to Mike 's house . 7 7 `` Chapter Seven : The Bathtub '' The Duffer Brothers Justin Doble July 15 , 2016 ( 2016 - 07 - 15 ) Lucas warns Mike via walkie - talkie that government agents are en route . Mike , Dustin , and Eleven flee . Eleven telekinetically flips a van that blocks their path , and the kids escape . Lucas reconciles with Mike and Eleven . Joyce and Hopper are called to the police station because of Jonathan 's assault . Jonathan and Nancy reveal their knowledge of the monster to Joyce and Hopper . The group contacts Mike and his friends and they rendezvous . They ask Eleven to search for Will and Barb , but her earlier feats have left her too weak to find them . To amplify Eleven 's powers , they break into the middle school and build a sensory - deprivation tank . Eleven finds Barb 's corpse with a slug - like creature crawling out of her mouth . With Joyce 's help , she finds Will alive , hiding in the Upside Down `` Castle Byers , '' his backyard fort . Hopper and Joyce break into Hawkins Laboratory but are apprehended by security guards . Nancy and Jonathan resolve to kill the monster and steal their hunting gear back from the police station . In the Upside Down , the monster breaks into the fort where Will is hiding . 8 8 `` Chapter Eight : The Upside Down '' The Duffer Brothers Story by : Paul Dichter Teleplay by : The Duffer Brothers July 15 , 2016 ( 2016 - 07 - 15 ) Hopper gives up Eleven 's location so that he and Joyce can enter the Upside Down . They find Will unconscious in the Upside - Down town library . They pull a tendril out of his throat , perform CPR , and revive him . Nancy and Jonathan booby - trap the Byers ' home , then cut their hands to attract the monster with their blood . Steve arrives , intending to apologize to Jonathan about their fight , but the monster attacks . The monster springs the trap but escapes back to the Upside Down . Eleven and the boys hide in the middle school . Mike asks Eleven to a school dance , the Snow Ball , and kisses her . The military storms the school to re-capture Eleven , but she kills them and collapses . As Brenner cradles her , the monster attacks , apparently killing him . The boys hide Eleven in a classroom . The monster corners them , but Eleven pins it against a wall . She says goodbye to Mike , disintegrates the monster , and vanishes . Will is hospitalized and reunited with his mother , brother , and friends . Hopper is reluctantly picked up by a black car . One month later , Nancy and Steve are back together , and both are friends with Jonathan . Hopper leaves Eggo waffles and other food in a box in the woods . Will coughs up a slug - like creature and has a vision of the world as the Upside Down , but he hides this from his family .
  Fact 0.0: The table lists, for each episode, its overall number, its number in the season, its title, its director, its writers, and its original release date.
  Fact 0.1: Chapter One: The Vanishing of Will Byers was directed by The Duffer Brothers and written by The Duffer Brothers.
  Fact 0.2: Chapter One: The Vanishing of Will Byers was released on July 15, 2016.
  Fact 0.3: On November 6, 1983, in a U.S. Department of Energy laboratory in the town of Hawkins, Indiana, a scientist is attacked by an unseen creature.
  Fact 0.4: While bicycling home from a Dungeons & Dragons session with his friends, 12-year-old Will Byers encounters the creature and vanishes.
  Fact 0.5: The next day, a buzz-cut young girl wearing a hospital gown steals food from a local diner.
  Fact 0.6: The diner's owner, Benny, takes pity on the girl and feeds her before calling social services.
  Fact 0.7: From a tattoo on her arm, Benny learns that the girl's name is Eleven.
  Fact 0.8: A woman posing as a social worker arrives and murders Benny.
  Fact 0.9: Armed men search the diner for Eleven, but she escapes.
  Fact 0.10: Will's mother Joyce believes she hears Will's voice on a distorted phone call, but her phone short circuits.
  Fact 0.11: Will's friends Mike, Dustin, and Lucas search for Will in the woods and find Eleven.
  Fact 0.12: Chapter Two: The Weirdo on Maple Street was directed by The Duffer Brothers and written by The Duffer Brothers.
  Fact 0.13: Chapter Two: The Weirdo on Maple Street was released on July 15, 2016.
  Fact 0.14: The boys bring Eleven to Mike's house, where she sleeps in the basement.
  Fact 0.15: The next day, Mike nicknames her "El".
  Fact 0.16: Eleven says that "bad people" are looking for her and refuses to meet Mike's parents.
  Fact 0.17: Scientists from the laboratory find a substance oozing from the walls of Joyce's home.
  Fact 0.18: At Mike's home, Eleven recognizes and points out Will in a photo.
  Fact 0.19: Dustin and Lucas want to inform Mike's parents about Eleven, but she uses psychokinesis to stop them.
  Fact 0.20: While searching for Will, Mr. Clarke, the boys' science teacher, discovers a scrap of a hospital gown outside the laboratory grounds.
  Fact 0.21: Nancy goes with her friend Barb to a party at her boyfriend Steve's house.
  Fact 0.22: Will's brother Jonathan investigates the woods where Will went missing.
  Fact 0.23: Hearing screaming, Jonathan runs to help but finds only Steve, Nancy, and their friends roughhousing around Steve's swimming pool.
  Fact 0.24: Jonathan secretly photographs Steve, Nancy, and their friends.
  Fact 0.25: Barb, left alone by the poolside, vanishes.
  Fact 0.26: Joyce receives another call from Will, hears music from his room, and sees something coming through the wall.
  Fact 0.27: Chapter Three: Holly, Jolly was directed by Shawn Levy and written by Jessica Mecklenburg.
  Fact 0.28: Chapter Three: Holly, Jolly was released on July 15, 2016.
  Fact 0.29: Barb wakes up in an empty and decaying pool.
  Fact 0.30: Barb screams for Nancy and attempts to climb out but is dragged down by something unseen.
  Fact 0.31: Joyce strings Christmas lights around her home to talk to Will, who can turn them on and off.
  Fact 0.32: Hawkins Lab director Dr. Martin Brenner allows Hopper to view the lab's security camera footage.
  Fact 0.33: Hopper realizes the footage is fake.
  Fact 0.34: Hopper and his deputies research Hawkins Lab, Brenner, and a woman named Terry Ives, who claimed her daughter was taken by Brenner.
  Fact 0.35: Eleven has a flashback in which Brenner, whom she calls "Papa," has her put in solitary confinement for refusing to telekinetically harm a cat.
  Fact 0.36: Nancy worries about Barb, who is missing.
  Fact 0.37: Steve and his friends discover Jonathan's photographs.
  Fact 0.38: Steve and his friends destroy Jonathan's photographs and his camera.
  Fact 0.39: Nancy recovers a photo of Barb by the poolside and returns to Steve's house to search for her.
  Fact 0.40: Nancy finds Barb's car and sees the creature in the woods.
  Fact 0.41: Joyce establishes a code with Will using the lights, which he uses to tell her that he is alive but unsafe.
  Fact 0.42: Will tells Joyce to run as a creature begins to climb through her wall.
  Fact 0.43: Will's apparent body is discovered in the water at a quarry.
  Fact 0.44: Chapter Four: The Body was directed by Shawn Levy and written by Justin Doble.
  Fact 0.45: Chapter Four: The Body was released on July 15, 2016.
  Fact 0.46: Eleven proves to the boys that Will is still alive by making contact through Mike's walkie-talkie.
  Fact 0.47: Joyce goes to the morgue to view the body and refuses to believe it is Will.
  Fact 0.48: The boys give Eleven a makeover so they can sneak her into school.
  Fact 0.49: Joyce hears Will in her living room wall.
  Fact 0.50: Tearing away the wallpaper, Joyce sees Will behind a membrane.
  Fact 0.51: At school, Eleven uses Mr. Clarke's ham radio to channel Will talking to his mother.
  Fact 0.52: Joyce demolishes the wall with an axe but reveals only her front porch.
  Fact 0.53: Nancy discovers a figure behind Barb in Jonathan's photo.
  Fact 0.54: Jonathan realizes the figure matches his mother's description of the creature: a humanoid figure with long arms and no face.
  Fact 0.55: Nancy tells the police about her missing friend.
  Fact 0.56: Nancy subsequently admits to her mother that she slept with Steve.
  Fact 0.57: When Nancy tells Steve about the police, she is irritated that he cares only about not getting in trouble with his father.
  Fact 0.58: Hopper confronts the state trooper who found Will's body and beats him until he admits he was ordered to lie.
  Fact 0.59: Hopper goes to the morgue and finds that Will's supposed body is a dummy.
  Fact 0.60: Hopper breaks into Hawkins laboratory.
  Fact 0.61: Chapter Five: The Flea and the Acrobat was directed by The Duffer Brothers and written by Alison Tatlock.
  Fact 0.62: Chapter Five: The Flea and the Acrobat was released on July 15, 2016.
  Fact 0.63: Will's father Lonnie assures Joyce her experiences are hallucinations.
  Fact 0.64: Hopper discovers a portal in the basement of the lab but is knocked out by the lab's guards.
  Fact 0.65: Hopper wakes at home and discovers a hidden microphone.
  Fact 0.66: Sheriff's deputies inform Hopper that Barb's car was found.
  Fact 0.67: Hopper finds the state's involvement suspicious.
  Fact 0.68: The boys determine Will is trapped in an alternate dimension that Eleven calls the "Upside Down."
  Fact 0.69: The boys determine Will is being pursued by a monster similar to a Demogorgon.
  Fact 0.70: After Will's funeral, science teacher Mr. Clarke tells the boys that a high-energy spacetime tear could create a passage between dimensions.
  Fact 0.71: Using their compasses, the boys search for a disruption in the Earth's electromagnetic field.
  Fact 0.72: Eleven remembers encountering the creature during a lab experiment.
  Fact 0.73: Afraid, Eleven redirects the compasses.
  Fact 0.74: Lucas notices and confronts Eleven.
  Fact 0.75: Mike defends Eleven, and he and Lucas fight.
  Fact 0.76: Eleven telekinetically flings Lucas off Mike.
  Fact 0.77: While Dustin and Mike tend to the unconscious Lucas, Eleven disappears.
  Fact 0.78: In the woods, Nancy and Jonathan find a wounded deer.
  Fact 0.79: The creature drags the deer away, and Nancy follows the trail of blood to a portal to the Upside Down.
  Fact 0.80: Nancy crawls through the portal and discovers the creature feasting upon the deer.
  Fact 0.81: Nancy snaps a twig and draws the creature's attention.
  Fact 0.82: Chapter Six: The Monster was directed by The Duffer Brothers and written by Jessie Nickson-Lopez.
  Fact 0.83: Chapter Six: The Monster was released on July 15, 2016.
  Fact 0.84: Jonathan pulls Nancy back through the portal and saves her from the monster.
  Fact 0.85: Later, Steve sees Nancy and Jonathan together through her bedroom window and assumes they are dating.
  Fact 0.86: Nancy and Jonathan resolve to kill the monster and purchase supplies from an army surplus store.
  Fact 0.87: During a fist fight with Steve, Jonathan inadvertently assaults a police officer and is arrested.
  Fact 0.88: Joyce and Hopper track down Terry Ives, who is catatonic and tended by her sister Becky.
  Fact 0.89: Becky explains that Terry underwent Project MKUltra testing while unknowingly pregnant.
  Fact 0.90: Becky explains that Terry believed her daughter Jane was kidnapped by Brenner at birth.
  Fact 0.91: Joyce and Hopper conclude Eleven is likely Jane.
  Fact 0.92: Eleven shoplifts Eggo waffles.
  Fact 0.93: Eleven recalls being asked to contact the monster and accidentally opening the portal to the Upside Down in the laboratory basement.
  Fact 0.94: While searching for Eleven, Mike and Dustin are ambushed by school bullies Troy and James.
  Fact 0.95: Troy holds Dustin at knifepoint and demands Mike make a suicidal jump into the lake where Will's body was discovered.
  Fact 0.96: Mike jumps but is levitated to safety by Eleven, who scares away the bullies.
  Fact 0.97: Lucas realizes agents from the laboratory are heading to Mike's house.
  Fact 0.98: Chapter Seven: The Bathtub was directed by The Duffer Brothers and written by Justin Doble.
  Fact 0.99: Chapter Seven: The Bathtub was released on July 15, 2016.
  Fact 0.100: Lucas warns Mike via walkie-talkie that government agents are en route.
  Fact 0.101: Mike, Dustin, and Eleven flee.
  Fact 0.102: Eleven telekinetically flips a van that blocks their path, and the kids escape.
  Fact 0.103: Lucas reconciles with Mike and Eleven.
  Fact 0.104: Joyce and Hopper are called to the police station because of Jonathan's assault.
  Fact 0.105: Jonathan and Nancy reveal their knowledge of the monster to Joyce and Hopper.
  Fact 0.106: The group contacts Mike and his friends, and they rendezvous.
  Fact 0.107: The group asks Eleven to search for Will and Barb, but her earlier feats have left her too weak to find them.
  Fact 0.108: To amplify Eleven's powers, the group breaks into the middle school and builds a sensory-deprivation tank.
  Fact 0.109: Eleven finds Barb's corpse with a slug-like creature crawling out of her mouth.
  Fact 0.110: With Joyce's help, Eleven finds Will alive, hiding in the Upside Down "Castle Byers," his backyard fort.
  Fact 0.111: Hopper and Joyce break into Hawkins Laboratory but are apprehended by security guards.
  Fact 0.112: Nancy and Jonathan resolve to kill the monster and steal their hunting gear back from the police station.
  Fact 0.113: In the Upside Down, the monster breaks into the fort where Will is hiding.
  Fact 0.114: Chapter Eight: The Upside Down was directed by The Duffer Brothers, with story by Paul Dichter and teleplay by The Duffer Brothers.
  Fact 0.115: Chapter Eight: The Upside Down was released on July 15, 2016.
  Fact 0.116: Hopper gives up Eleven's location so that he and Joyce can enter the Upside Down.
  Fact 0.117: Hopper and Joyce find Will unconscious in the Upside-Down town library.
  Fact 0.118: Hopper and Joyce pull a tendril out of Will's throat, perform CPR, and revive him.
  Fact 0.119: Nancy and Jonathan booby-trap the Byers' home, then cut their hands to attract the monster with their blood.
  Fact 0.120: Steve arrives, intending to apologize to Jonathan about their fight, but the monster attacks.
  Fact 0.121: The monster springs the trap but escapes back to the Upside Down.
  Fact 0.122: Eleven and the boys hide in the middle school.
  Fact 0.123: Mike asks Eleven to a school dance, the Snow Ball, and kisses her.
  Fact 0.124: The military storms the school to re-capture Eleven, but she kills them and collapses.
  Fact 0.125: As Brenner cradles Eleven, the monster attacks, apparently killing him.
  Fact 0.126: The boys hide Eleven in a classroom.
  Fact 0.127: The monster corners the boys and Eleven, but Eleven pins it against a wall.
  Fact 0.128: Eleven says goodbye to Mike, disintegrates the monster, and vanishes.
  Fact 0.129: Will is hospitalized and reunited with his mother, brother, and friends.
  Fact 0.130: Hopper is reluctantly picked up by a black car.
  Fact 0.131: One month later, Nancy and Steve are back together, and both are friends with Jonathan.
  Fact 0.132: Hopper leaves Eggo waffles and other food in a box in the woods.
  Fact 0.133: Will coughs up a slug-like creature and has a vision of the world as the Upside Down, but he hides this from his family.

Retrieved items:
<item 1>
Stranger Things
Her wrist tattoo marks her as `` 008 . '' At the Hawkins arcade , Will has a vision of the Upside Down . Murray Bauman describes an elaborate conspiracy theory
to Hopper . Hopper investigates a blight on Bob Merril 's pumpkin patch . Mr. Clarke introduces a new girl named Maxine ( `` Max '' ) to his class . At Hawkins
Lab , Owens examines Will . He tells Joyce that Will is experiencing posttraumatic stress disorder . Nancy and Steve have dinner with Barb 's parents , who are
selling their house to pay for Bauman 's investigation into Barb 's disappearance . Mike tries to contact with Eleven , still believing she is alive even 352
days after her disappearance . Dustin hears a noise from the trashcan outside his home , and Will has a vision of an enormous spidery creature in the Upside
Down . Hopper goes home to a cabin in the woods , where Eleven accuses him of being late .
</item 1>

<item 2>
Stranger Things
Episodes Season 1 ( 2016 ) No . overall No. in season Title Directed by Written by Original release date `` Chapter One : The Vanishing of Will Byers '' The
Duffer Brothers The Duffer Brothers July 15 , 2016 ( 2016 - 07 - 15 ) On November 6 , 1983 , in a U.S. Department of Energy laboratory in the town of Hawkins ,
Indiana , a scientist is attacked by an unseen creature . While bicycling home from a Dungeons & Dragons session with his friends , 12 - year - old Will Byers
encounters the creature and vanishes . The next day , a buzz - cut young girl wearing a hospital gown steals food from a local diner . The owner , Benny , takes
pity on her and feeds her before calling social services . From a tattoo on her arm , he learns that her name is Eleven . A woman posing as a social worker
arrives and murders Benny . Armed men search the diner for Eleven , but she escapes . Will 's mother Joyce believes she hears Will 's voice on a distorted phone
call , but her phone short circuits .
</item 2>

<item 3>
Stranger Things
Cast and characters The Duffer Brothers , Shawn Levy and the main cast of season 2 of Stranger Things at the 2017 San Diego Comic - Con International Main
Winona Ryder as Joyce Byers , the mother of Will and Jonathan Byers . She is divorced from Lonnie Byers . David Harbour as Jim Hopper , chief of Hawkins Police
Department . After his young daughter died of cancer , Hopper got divorced and lapsed into alcoholism . Finn Wolfhard as Mike Wheeler , son of Karen and Ted ,
younger brother of Nancy , and one of three friends of Will Byers . He is an intelligent and conscientious student and is committed to his friends . He develops
romantic feelings for Eleven . Millie Bobby Brown as Eleven ( `` El '' ) , a young girl with psychokinetic abilities and a limited vocabulary . Her real name is
Jane , the biological daughter of Terry Ives . After escaping from Hawkins Laboratory , where experiments were being performed on her , she befriends Mike ,
Dustin , and Lucas . She develops romantic feelings for Mike . Gaten Matarazzo as Dustin Henderson , one of Will 's friends .
</item 3>

<item 4>
Stranger Things
As Brenner cradles her , the monster attacks , apparently killing him . The boys hide Eleven in a classroom . The monster corners them , but Eleven pins it
against a wall . She says goodbye to Mike , disintegrates the monster , and vanishes . Will is hospitalized and reunited with his mother , brother , and friends
. Hopper is reluctantly picked up by a black car . One month later , Nancy and Steve are back together , and both are friends with Jonathan . Hopper leaves Eggo
waffles and other food in a box in the woods . Will coughs up a slug - like creature and has a vision of the world as the Upside Down , but he hides this from
his family . Season 2 ( 2017 ) No . overall No. in season Title Directed by Written by Original release date 9 `` Chapter One : MADMAX '' The Duffer Brothers
The Duffer Brothers October 27 , 2017 ( 2017 - 10 - 27 ) On October 28 , 1984 , Kali and her gang use her psychic powers to escape the police . Her wrist tattoo
marks her as `` 008 . ''
</item 4>

<item 5>
Stranger Things
He breaks into Hawkins laboratory . 5 5 `` Chapter Five : The Flea and the Acrobat '' The Duffer Brothers Alison Tatlock July 15 , 2016 ( 2016 - 07 - 15 ) Will
's father Lonnie assures Joyce her experiences are hallucinations . Hopper discovers a portal in the basement of the lab but is knocked out by the lab 's guards
. He wakes at home and discovers a hidden microphone . Sheriff 's deputies inform Hopper that Barb 's car was found , but Hopper finds the state 's involvement
suspicious . The boys determine Will is trapped in an alternate dimension Eleven calls the `` Upside Down , '' and he is being pursued by a monster similar to a
Demogorgon . After Will 's funeral , science teacher Mr. Clarke tells the boys that a high - energy spacetime tear could create a passage between dimensions .
Using their compasses , they search for a disruption in the Earth 's electromagnetic field . Eleven remembers encountering the creature during a lab experiment
, and afraid , she redirects the compasses . Lucas notices and confronts her .
</item 5>

<item 6>
Stranger Things
The boys give Eleven a makeover so they can sneak her into school . Joyce hears Will in her living room wall . Tearing away the wallpaper , she sees him behind
a membrane . At school , Eleven uses Mr. Clarke 's ham radio to channel Will talking to his mother . Joyce demolishes the wall with an axe but reveals only her
front porch . Nancy discovers a figure behind Barb in Jonathan 's photo . Jonathan realizes that the figure matches his mother 's description of the creature :
a humanoid figure with long arms and no face . Nancy tells the police about her missing friend and subsequently admits to her mother that she slept with Steve .
When she tells Steve about the police , she is irritated that he cares only about not getting in trouble with his father . Hopper confronts the state trooper
who found Will 's body and beats him until he admits he was ordered to lie . Hopper goes to the morgue and finds that Will 's supposed body is a dummy . He
breaks into Hawkins laboratory .
</item 6>

<item 7>
Stranger Things
Gaten Matarazzo as Dustin Henderson , one of Will 's friends . His cleidocranial dysplasia causes him to lisp . In the second season , he is proud of his new
front teeth and is attracted to Max . Caleb McLaughlin as Lucas Sinclair , one of Will 's friends . He is wary of Eleven but later befriends her . In season two
, he is one of Max 's love interests . Natalia Dyer as Nancy Wheeler , daughter of Karen and Ted and older sister of Mike . In season one , she is Steve
Harrington 's girlfriend . In season two , she becomes Jonathan Byers ' girlfriend . Charlie Heaton as Jonathan Byers , the older brother of Will Byers and the
son of Joyce Byers . He is a quiet teenager , an outsider at school , and an aspiring photographer . He is close with his mother and brother , and he becomes
the boyfriend of Nancy Wheeler . Cara Buono as Karen Wheeler , mother of teenaged Nancy , middle schooler Mike , and toddler Holly . Matthew Modine as Martin
Brenner , the scientist in charge of Hawkins Laboratory .
</item 7>

<item 8>
Stranger Things
Eleven has a flashback in which Brenner , whom she calls `` Papa , '' has her put in solitary confinement for refusing to telekinetically harm a cat . Nancy
worries about Barb , who is missing . Steve and his friends discover Jonathan 's photographs . They destroy them and his camera . Nancy recovers a photo of Barb
by the poolside and returns to Steve 's house to search for her . Nancy finds Barb 's car and sees the creature in the woods . Joyce establishes a code with
Will using the lights , which he uses to tell her that he is alive but unsafe . He tells Joyce to run as a creature begins to climb through her wall . Will 's
apparent body is discovered in the water at a quarry . `` Chapter Four : The Body '' Shawn Levy Justin Doble July 15 , 2016 ( 2016 - 07 - 15 ) Eleven proves to
the boys that Will is still alive by making contact through Mike 's walkie - talkie . Joyce goes to the morgue to view the body and refuses to believe it is
Will . The boys give Eleven a makeover so they can sneak her into school .
</item 8>

<item 9>
Stranger Things
Joe Davison as a technician at Hawkins Laboratory . Priah Ferguson as Erica Sinclair , Lucas ' younger sister . Guest Introduced in season one Ross Partridge as
Lonnie Byers , the drunk , egotistical , selfish ex-husband of Joyce Byers and father of Jonathan and Will . He has a much younger girlfriend named Cynthia .
Chris Sullivan as Benny Hammond , owner and chef of Benny 's Burgers , and friend of Hopper . Andrew Benator as a scientist at Hawkins Laboratory . Glennellen
Anderson as Nicole , a classmate of Nancy , Steve and Jonathan . Pete Burris as the head of security at Hawkins Laboratory . Jerri Tubbs as Diane Hopper , Jim
's ex-wife . Amy Seimetz as Becky Ives , Terry 's sister who takes care of her . Elle Graham as Sarah Hopper , Jim and Diane 's daughter . Executive producer
Shawn Levy cameos as a morgue worker . Introduced in season two Kai L. Greene as Funshine , a member of Kali 's crew . James Landry Hébert as Axel , a member of
Kali 's crew .
</item 9>

<item 10>
Stranger Things
Matthew Modine as Martin Brenner , the scientist in charge of Hawkins Laboratory . He is manipulative and remote . He and his team are searching for Eleven . He
reappears in season 2 briefly . Noah Schnapp as Will Byers , the younger brother of Jonathan Byers and the son of Joyce Byers . He is captured by a monster from
the `` Upside Down '' , an alternate dimension discovered by Hawkins Laboratory scientists . Schnapp was promoted to series regular for the second season ,
after recurring in the first . Sadie Sink as Maxine `` Max '' Mayfield / `` Madmax '' , Billy 's younger stepsister , and a tomboy who catches the attention of
both Lucas and Dustin . Joe Keery as Steve Harrington , a popular high school student and the boyfriend of Nancy Wheeler . He ostracizes Jonathan Byers , but
later comes to befriend him . He begins as Nancy 's boyfriend but breaks up with her in season two . Keery was promoted to series regular for the second season
, after recurring in the first . Dacre Montgomery as Billy Hargrove , Max 's violent and unpredictable older stepbrother .
</item 10>
=== END ITEM id=f4c684a05f8c592c ===

=== ITEM id=7c46e6bd79e3ee7b ===
Gold sentences and facts:
Sentence 0: Health or vitality is an attribute assigned to entities , such as the player character , enemies and objects within a role - playing or video game , that indicates its state in combat . Health is usually measured in hit points or health points , shortened to HP . When the HP of a player character reaches zero , the player may lose a life or their character might become incapacitated or die . When the HP of an enemy reaches zero , it may be defeated or die and the player is usually rewarded in some way .
  Fact 0.0: Health or vitality is an attribute assigned to entities in a role-playing or video game.
  Fact 0.1: Health or vitality is assigned to entities such as the player character, enemies, and objects.
  Fact 0.2: Health or vitality indicates the state of an entity in combat.
  Fact 0.3: Health is usually measured in hit points or health points.
  Fact 0.4: Hit points or health points are shortened to HP.
  Fact 0.5: When the HP of a player character reaches zero, the player may lose a life.
  Fact 0.6: When the HP of a player character reaches zero, the character might become incapacitated or die.
  Fact 0.7: When the HP of an enemy reaches zero, the enemy may be defeated or die.
  Fact 0.8: When the HP of an enemy reaches zero, the player is usually rewarded in some way.

Retrieved items:
<item 1>
Health (gaming)
wikipedia.org/w/index.php?title=Health_(gaming)&oldid=819315199 '' Categories : Role - playing game terminology Video game terminology Hidden categories : All
articles lacking reliable references Articles lacking reliable references from September 2016 Talk Contents About Wikipedia Deutsch Español فارسی Français 한국어
Italiano עברית ქართული Nederlands 日本 語 Norsk Norsk nynorsk Polski Português Русский Suomi Svenska Українська 中文 Edit links This page was last edited on 8
January 2018 , at 18 : 11 . Text is available under the Creative Commons Attribution - ShareAlike License ; additional terms may apply . By using this site ,
you agree to the Terms of Use and Privacy Policy . Wikipedia ® is a registered trademark of the Wikimedia Foundation , Inc. , a non-profit organization . About
Wikipedia
</item 1>

<item 2>
Health (gaming)
`` Disposable Bodies : Cyborg Regeneration and FPS Mechanics '' . Guns , Grenades , and Grunts : First - Person Shooter Games ( Google eBook ) . Bloomsbury
Publishing USA . ISBN 1441191445 . Retrieved 2014 - 11 - 21 . ( hide ) Concepts of video games Glossary of video game terms Attributes Health Life Experience
point Magic Mechanics Item Power - up HUD Warp Fog of war Invisible wall Cutscene Mini-map Paper doll Destructible environment Loading screen Replay value
Scripted sequence Password Line of sight Critical hit Level Permadeath Status effect Hitbox Movement techniques Rocket jumping Strafe - jumping Strafing
Trickjump Modes Player type Single - player Multiplayer Cooperative Competitive gameplay eSports Speedrun Gameplay type Capture the flag ( CTF ) Deathmatch Last
man standing Retrieved from `` https://en.wikipedia.org/w/index.
</item 2>

<item 3>
Health (gaming)
US Navy used similar concept in their tactical war games already in 1920s and 1930s . In their simulation each ship had Life parameter . The unit of Life of the
ship was a number of `` equivalent penetrative 14 - inch shell hits '' . The Navy considered ie. that Kongō - class battlecruiser had 12 Life points and Nagato
- class battleship had 18.8 . A visual power meter representing stamina was used in Nintendo 's 1983 arcade game Punch - Out ! ! and Data East 's 1981 DECO
Cassette System arcade game Flash Boy . Usage ( edit ) In action video games as well as in role - playing games , health points can usually be depleted by
attacking the entity . A defense attribute might reduce the amount of HP that is lost when a character is damaged . It is common in role - playing games for a
character 's maximum health and defense attributes to be gradually raised as the character levels up . In game design , it is deemed important that a player is
aware of it when they are losing health , each hit playing a clear sound effect .
</item 3>

<item 4>
Health (gaming)
This mechanic initially appeared in action role - playing games , with early examples including the Hydlide series , the Ys series , and Woody Poco . In Woody
Poco , the rate at which health recharges is based on food level . In Hydlide and Ys , the player character has to stand still for their health to automatically
regenerate . This system was popularized in first - person shooters by Halo : Combat Evolved ( 2001 ) , though regenerating health in The Getaway ( 2002 ) has
been cited to be more comparable to later use of the mechanic in first - person shooters . Display ( edit ) Heart - shaped icons can indicate the amount of
health a player has left . The way health is displayed on the screen has an effect on the player . Many games only show the health of the player character ,
while keeping the health of enemies hidden . This is done in the Legend of Zelda series , Minecraft and Monster Hunter series to keep the player 's progress in
defeating their enemy unclear and therefore exciting . In these games , the fact that the enemies are being damaged is indicated by their behavior .
</item 4>

<item 5>
Health (gaming)
Health ( Gaming ) - wikipedia Health ( Gaming ) Jump to : navigation , search A health bar , a possible representation of the health of a character . Health or
vitality is an attribute assigned to entities , such as the player character , enemies and objects within a role - playing or video game , that indicates its
state in combat . Health is usually measured in hit points or health points , shortened to HP . When the HP of a player character reaches zero , the player may
lose a life or their character might become incapacitated or die . When the HP of an enemy reaches zero , it may be defeated or die and the player is usually
rewarded in some way . Any entity within a game could have a health value , including the player character , non-player characters and objects . Indestructible
entities have no diminishable health value . Health might be displayed as a numeric value , such as `` 50 / 100 '' . Here , the first number indicates the
current amount of HP an entity has and the second number indicates the entity 's maximum HP .
</item 5>

<item 6>
Health (gaming)
In these games , the fact that the enemies are being damaged is indicated by their behavior . On the other hand , many fighting games , such as the Street
Fighter series , use easy - to - read health bars to clearly indicate the progress the player is making with each hit . It is common in first - person shooters
to indicate low health of the player character by blood spatters or by a distorted red hue on the screen , attempting to mimic the effects of wounding and
trauma . These visual effects fade as health regenerates . References ( edit ) ^ Jump up to : Moore , Michael ( 2011 - 03 - 23 ) . Basics of Game Design . CRC
Press . pp. 151 , 194 . ISBN 1439867763 . Retrieved 2014 - 12 - 09 . Jump up ^ Chris Antista . `` The 10 most creative life bars '' . GamesRadar. 2010 - 08 - 17
. p. 2 . Archived from the original on 2014 - 12 - 28 .
</item 6>

<item 7>
Health (gaming)
In video games , health can also be displayed graphically , such as with a bar that empties itself when an entity loses health ( a health bar , typically red )
, icons that are `` chipped away '' , or in more novel ways . Contents ( hide ) 1 History 2 Usage 2.1 Regeneration 3 Display 4 References History ( edit )
Dungeons & Dragons co-creator Dave Arneson described the origin of hit points in a 2002 interview . When Arneson was adapting the medieval wargame Chainmail (
1971 ) to a fantasy setting , a process that with Gary Gygax would lead to Dungeons & Dragons , he saw that the emphasis of the gameplay was moving from large
armies to small groups of heroes and eventually to the identification of one player and one character that is essential to role - playing as it was originally
conceived . Players became attached to their heroes and did not want them to die every time they lost a die roll . Players were thus given multiple hit points
which were incrementally decreased as they took damage . Arneson took the concept , along with armor class , from a set of a naval American Civil War game 's
rules .
</item 7>

<item 8>
Health (gaming)
`` Falcom : Legacy of Ys '' . Games ( 111 ) : 152 -- 159 ( 153 ) . ( cf . Szczepaniak , John ( July 8 , 2011 ) . `` History of Ys interviews '' . Hardcore
Gaming 101 . Retrieved 6 September 2011. ) Jump up ^ Jon Martindale ( 2012 - 10 - 03 ) . `` Let 's Kill off Health Bars '' . Kit Guru Gaming . Archived from the
original on May 28 , 2015 . Retrieved 2014 - 11 - 21 . Jump up ^ Novak , Jeannie ( 2013 - 04 - 11 ) . The Official GameSalad Guide to Game Development . Cengage
Learning . p. 31 . ISBN 1133605648 . Retrieved 2014 - 11 - 21 . Jump up ^ Voorhees , Gerald A. ; Call , Joshua ; Whitlock , Katie ( 2012 - 11 - 02 ) .
</item 8>

<item 9>
Health (gaming)
Author Scott Rogers states that `` health should deplete in an obvious manner , because with every hit , a player is closer to losing their life . '' The
display of health also helps to dramatize the near - loss of a life . Regeneration ( edit ) Player characters can often restore their health points by consuming
certain items , such as health potions , food or first - aid kits . Staying a night at an inn fully restores a character 's health in many role - playing video
games . In general , the different methods of regenerating health has its uses in a particular genre . In action games , this method is very quick , whereas
role - playing games feature slower paced methods to match the gameplay and realism . Some video games feature automatically regenerating health , where lost
health points are regained over time . This can be useful to not `` cripple '' the player , making them still able to continue even after losing lots of health
. However , automatically regenerating health may also cause a player to `` power through '' sections they might otherwise have had to approach cautiously ,
simply because there are no lasting consequences to losing a large amount of health .
</item 9>

<item 10>
Health (gaming)
Jump up ^ Jonathan Moriarty ( 2010 - 12 - 02 ) . `` Video Game Basics : The Health Bar '' . Baltimoregamer.com . Archived from the original on 28 April 2012 .
Retrieved 2014 - 11 - 21 . ^ Jump up to : Dunn , Jeff ( 2012 - 11 - 15 ) . `` Stop , Drop , and Heal : The history of regenerating health '' . GamesRadar .
Retrieved 2015 - 01 - 08 . Jump up ^ Sulliven , Lucas . `` Top 7 ... Games you did n't know did it first '' . GamesRadar . Retrieved 2015 - 01 - 08 . ^ Jump up
to : John Szczepaniak ( 2016 ) . `` dB - SOFT Gaming 101 '' . The Untold History of Japanese Game Developers . 2 . Jump up ^ Szczepaniak , John ( 7 July 2011 )
. `` Falcom : Legacy of Ys '' .
</item 10>
=== END ITEM id=7c46e6bd79e3ee7b ===

=== ITEM id=e08b50fe66bf6392 ===
Gold sentences and facts:
Sentence 0: Republic of Doyle is a Canadian comedy - drama television series set in St. John 's , Newfoundland and Labrador which aired on CBC Television from January 6 , 2010 to December 10 , 2014 .
  Fact 0.0: Republic of Doyle is a Canadian comedy-drama television series.
  Fact 0.1: Republic of Doyle is set in St. John's, Newfoundland and Labrador.
  Fact 0.2: Republic of Doyle aired on CBC Television.
  Fact 0.3: Republic of Doyle aired from January 6, 2010 to December 10, 2014.

Retrieved items:
<item 1>
Republic of Doyle
Jump up ^ `` Republic of Doyle , Season 2 '' . Amazon.com . Retrieved 2017 - 10 - 04 . Jump up ^ `` REPUBLIC OF DOYLE - SEASON 1 Television Series Drama Comedy
'' . Beyondhomeentertainment.com.au . Retrieved 2012 - 11 - 09 . Jump up ^ `` REPUBLIC OF DOYLE - SEASON 2 Television Series Drama Comedy '' .
Beyondhomeentertainment.com.au . Retrieved 2012 - 11 - 09 . Jump up ^ `` Republic of Doyle '' . TVSA. 2010 - 08 - 01 . Retrieved 2010 - 08 - 01 . Jump up ^ ``
UKTV enters Republic of Doyle '' ( Press release ) . C21 Media . 2010 - 08 - 26 . Retrieved 2010 - 08 - 26 . Jump up ^ `` ABC2 Programming Airdate : Republic of
Doyle ( episode one ) '' . ABC Television Publicity. 2010 - 11 - 20 . Jump up ^ Newsome , Brad ( 6 January 2012 ) .
</item 1>

<item 2>
Republic of Doyle
2015 TBA The Complete Series 77 September 6 , 2016 TBA International broadcasters ( edit ) Country Channel Premiere date South Africa Fox Entertainment 1 August
2010 United Kingdom Alibi 27 October 2010 Australia ABC2 ( season one ) 13th Street ( season two -- ) 6 November 2010 Brazil Globosat HD 11 November 2010
Estonia Fox Crime 11 September 2010 Serbia Fox Crime 14 October 2010 Bulgaria Fox Crime 4 December 2010 Poland TVP2 25 December 2012 Germany 13th Street 16
October 2013 United States CW Every Saturday and Sunday Netherlands 13th Street Until Friday 13 March 2015 , daily Monday - Friday Republic of Doyle is aired in
the USA on various stations via syndication . References ( edit ) Jump up ^ `` CBC winter season brings 3 new shows '' . cbc.ca , November 25 , 2009 . Jump up ^
`` Full Cast & Crew . '' IMDb . IMDb.com , n.d. Web . 17 Feb. 2014 . Jump up ^ `` Republic of Doyle renewed for 6th season '' .
</item 2>

<item 3>
Republic of Doyle
The character name is evocative of the character name , Bobby Crocker , and the actor who played him , Kevin Dobson , in the classic television series Kojak (
1973 -- 78 ) , starring Telly Savalas . Production ( edit ) The show is filmed in and around St. John 's , Newfoundland and Labrador . The show 's star , Allan
Hawco , is also executive producer / showrunner / co-creator and head writer of the show . Other producers include John Vatcher ( exec producer ) , Rob Blackie
, Perry Chafe ( co-creator and writer ) and Michael Levine . In 2013 , Republic of Doyle aired a fictional crossover with another CBC Television crime drama ,
Murdoch Mysteries . With the two shows set over 100 years apart , Hawco appeared on the November 25 , 2013 episode of Murdoch Mysteries as Jacob Doyle , a 19th
- century ancestor of Jake Doyle , while Yannick Bisson appeared on the January 29 , 2014 episode of Republic of Doyle as Detective Bill Murdoch , a 21st -
century descendant of his regular character Detective William Murdoch .
</item 3>

<item 4>
Republic of Doyle
After being hit by a car in a parking garage , the medication used to mend Jake 's injuries from the incident , gives him some peculiar dreams . The season
comes to an end with Crocker and Jake being forced to collaborate in order to save the kidnapped Tinny . Season Five ( edit ) After the wild adventure Jake took
in the previous season by saving Tinny , his rescue mission leads to his kidnapping . The Doyles have been anxiously searching for him for the past two months
and Tinny risks her career as a police officer and becomes unprofessional by involving her personal life in her police work and helping Jake . As a result ,
Tinny gets suspended . Meanwhile , Des overcomes his sorrow over losing Tinny . Jake experiences more complications in returning to St. John 's , but eventually
makes it home safe with the help of his niece . Jake and Leslie 's relationship heats up with Jake nearly impregnating her . When Leslie is being followed and
nearly gets shot , Jake is extremely worried for her safety but soon finds out her pursuer is actually her long - lost husband . Jake and Leslie try but fail to
smooth their relationship with the shocking news .
</item 4>

<item 5>
Republic of Doyle
John 's , Newfoundland and Labrador Irish - Canadian culture Television shows set in Newfoundland and Labrador Television series produced in Newfoundland and
Labrador 2014 Canadian television series endings 2010s Canadian comedy - drama television series Television series by Entertainment One Fictional Canadian
people of Irish descent Hidden categories : Webarchive template archiveis links Talk Contents About Wikipedia Deutsch Français Polski Português Edit links This
page was last edited on 23 April 2018 , at 17 : 01 . About Wikipedia
</item 5>

<item 6>
Republic of Doyle
Republic of Doyle - wikipedia Republic of Doyle Jump to : navigation , search Republic of Doyle Genre Comedy - drama Police procedural Created by Allan Hawco
Malcolm MacRury Perry Chafe Starring Allan Hawco Seán McGinley Lynda Boyd Krystin Pellerin Mark O'Brien Marthe Bernard Bob Cole Opening theme `` Oh Yeah '' by
Great Big Sea Original language ( s ) English No. of seasons 6 No. of episodes 77 ( list of episodes ) Production Executive producer ( s ) Allan Hawco , John
Vatcher , Michael Levine Location ( s ) St. John 's , Newfoundland and Labrador , Canada Cinematography Malcolm Cross Running time 44 minutes Production company
( s ) Take the Shot Productions Fireworks ( seasons 1 - 2 ) Content Film ( season 3 - 6 ) Distributor Trifecta Entertainment & Media ( United States ) Release
Original network CBC Television Picture format 1080i ( HDTV ) Original release January 6 , 2010 -- December 10 , 2014 External links Website Republic of Doyle
is a Canadian comedy - drama television series set in St.
</item 6>

<item 7>
Republic of Doyle
Dramatic Series Gordon Pinsent Won 2014 Best Cross-Platform Project , Fiction - The Republic of Doyle Ride - Along App Allen Martin , Eva Riinitze , Fergus
Heywood , Helen Asimakis , John Vatcher , Mike Evans , Rose Paton Nominated Gemini Awards ( edit ) Year Category Nominee Result Ref Best Dramatic Series Allan
Hawco , John Vatcher , Rob Blackie , Michael Levine Nominated Best Performance by an Actor in a Continuing Leading Dramatic Role Allan Hawco Nominated Best
Performance by an Actress in a Continuing Leading Dramatic Role Lynda Boyd Nominated Best Performance by an Actress in a Featured Supporting Role in a Dramatic
Series Rachel Wilson Nominated Best Photography in a Dramatic Program or Series Michael Storey Nominated 2011 Best Performance by an Actress in a Continuing
Leading Dramatic Role Krystin Pellerin Nominated Best Production Design or Art Direction in a Fiction Program or Series Gordon Barnes Nominated Best Sound in a
Dramatic Series Scott Yates , Peter Clements , Harvey Hyslop , Andrew Tay , Paul Steffler , Mark Shnuriwsky , Lori Clarke Nominated DVD releases ( edit )
Entertainment One has released all six seasons on DVD in Canada .
</item 7>

<item 8>
Republic of Doyle
Courtney ( Seasons 1 -- 6 ) Marthe Bernard -- Katrina `` Tinny '' Doyle ( Seasons 1 -- 6 ) Bob Cole -- Voice of the Republic ( Seasons 1 -- 6 ) Recurring ( edit
) Steve O'Connell -- Sergeant Daniel Hood ( Seasons 1 -- 6 ) Sean Panting -- Walter McLean ( Seasons 1 -- 6 ) Michelle Nolden -- Allison Jenkins ( Seasons 2 --
5 ) Jonathan Goad -- Christian Doyle ( Seasons 1 -- 6 ) Krista Bridges - Kathleen Doyle ( Seasons 3 - 6 ) Lola Tash -- Sloan Daniels ( appears in 12 episodes
from Seasons 5 -- 6 ) Rick Roberts -- Mayor William Cadigan Clarke ( Seasons 2 -- 5 ) Jonathan Keltz -- Grayson Mann ( Seasons 4 -- 5 ) Patricia Isaac -- Monica
Hayward ( Season 4 ) Nicholas Campbell -- Martin Poole ( Seasons 1 -- 5 ) Paul Gross -- Kevin Crocker ( Seasons 2 -- 5 ) Gordon Pinsent -- Maurice Becker (
Seasons 1 -- 4 ) Victor Garber -- Garrison Steele ( Seasons 1 -- 5 ) Alan Doyle -- Wolf Redmond ( Seasons 2 --
</item 8>

<item 9>
Republic of Doyle
John 's , Newfoundland and Labrador which aired on CBC Television from January 6 , 2010 to December 10 , 2014 . The show stars Allan Hawco as private
investigator and former police officer Jake Doyle and Seán McGinley as his father , retired police officer Malachy Doyle . They partner as private investigators
with Rose Doyle , Malachy 's second wife , played by Lynda Boyd , in St. John 's . Krystin Pellerin stars as Constable / Sergeant Leslie Bennett , with Mark
O'Brien as Desmond `` Des '' Courtney , who works with the Doyles , and Marthe Bernard as Katrina `` Tinny '' Doyle , Jake 's niece and Malachy 's granddaughter
. Their cases involve them in all sorts of dealings -- not all of them on the right side of the law . The show was renewed on April 4 , 2014 , for a sixth and
final season .
</item 9>

<item 10>
Republic of Doyle
Jump up ^ Newsome , Brad ( 6 January 2012 ) . `` Republic of Doyle , Friday , January 6 '' . The Sydney Morning Herald . Retrieved 17 January 2012 . Jump up ^
Daniel MacEachern ( 2013 - 04 - 17 ) . `` ' Republic of Doyle ' headed for U.S. '' The Telegram . Retrieved 2014 - 11 - 21 . External links ( edit ) Television
in Canada portal Official website Republic of Doyle on IMDb Retrieved from `` https://en.wikipedia.org/w/index.php?title=Republic_of_Doyle&oldid=837889899 ''
Categories : 2010 Canadian television series debuts CBC Television shows Culture of St.
</item 10>
=== END ITEM id=e08b50fe66bf6392 ===
