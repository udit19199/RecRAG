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

=== ITEM id=fe15c7c9d2988f8d ===
Gold sentences and facts:
Sentence 0: `` For What It 's Worth ( Stop , Hey What 's That Sound ) '' ( often referred to as simply `` For What It 's Worth '' ) is a song written by Stephen Stills . It was performed by Buffalo Springfield , recorded on December 5 , 1966 , and released as a single on Atco Records in January 1967 . The single peaked at number seven on the Billboard Hot 100 chart . This song is currently ranked number 63 on Rolling Stone 's list of The 500 Greatest Songs of All Time as well as the eighth best song of 1967 by Acclaimed Music .
  Fact 0.0: 'For What It's Worth (Stop, Hey What's That Sound)' is often referred to as simply 'For What It's Worth'.
  Fact 0.1: 'For What It's Worth' is a song.
  Fact 0.2: 'For What It's Worth' was written by Stephen Stills.
  Fact 0.3: 'For What It's Worth' was performed by Buffalo Springfield.
  Fact 0.4: 'For What It's Worth' was recorded on December 5, 1966.
  Fact 0.5: 'For What It's Worth' was released as a single on Atco Records in January 1967.
  Fact 0.6: The 'For What It's Worth' single peaked at number seven on the Billboard Hot 100 chart.
  Fact 0.7: 'For What It's Worth' is currently ranked number 63 on Rolling Stone's list of The 500 Greatest Songs of All Time.
  Fact 0.8: 'For What It's Worth' is the eighth best song of 1967 according to Acclaimed Music.

Retrieved items:
<item 1>
For What It's Worth
Tour 2011 Retrieved from `` https://en.wikipedia.org/w/index.php?title=For_What_It%27s_Worth&oldid=815118566 '' Categories : 1966 songs 1967 singles 1969
singles Atco Records singles Buffalo Springfield songs Cher songs Counterculture of the 1960s Grammy Hall of Fame Award recipients Protest songs Songs about
California Songs written by Stephen Stills Song recordings produced by Arif Mardin Song recordings produced by Tom Dowd Song recordings produced by Jerry Wexler
Hidden categories : Articles with hAudio microformats Talk Contents About Wikipedia Čeština Español Français Italiano Nederlands Norsk nynorsk Português Русский
Srpskohrvatski / српскохрватски Українська Edit links This page was last edited on 12 December 2017 , at 21 : 49 . Text is available under the Creative Commons
Attribution - ShareAlike License ; additional terms may apply . By using this site , you agree to the Terms of Use and Privacy Policy . Wikipedia ® is a
registered trademark of the Wikimedia Foundation , Inc.
</item 1>

<item 2>
For What It's Worth
For What It 's Worth - wikipedia For What It 's Worth Jump to : navigation , search This article is about the Stephen Stills song . For other uses , see For
What It 's Worth ( disambiguation ) . `` What 's that sound ? '' redirects here . For the album , see What 's That Sound ? `` For What It 's Worth '' Single by
Buffalo Springfield from the album Buffalo Springfield B - side `` Do I Have to Come Right Out and Say It ? ''
</item 2>

<item 3>
For What It's Worth
Thompson summed up his review by saying , `` Rock history has sidelined Cassidy with a finality that seems almost personal , shovelling him up a teen idol side
street from which the only escape was retirement and resignation . Cassidy Live , however , was recorded while he still thought he had a chance of advancing --
and few period live albums come even close to capturing its excitement . '' Public Enemy sampled `` For What It 's Worth '' , for their 1998 song `` He Got Game
'' , which also featured Stephen Stills . Oui 3 adapted the song for their 1993 debut single of the same name , which reached number 26 in the UK chart . In
2017 , Haley Reinhart released a cover of the song as the third single from her third studio album , What 's That Sound ? References ( edit ) Jump up ^
Unterberger , Richie . `` Great Moments in Folk Rock : Lists of Author Favorites '' . Richieunterberger.com . Retrieved 2011 - 01 - 26 . Jump up ^ Fontenot ,
Robert . `` What is Country Rock ? '' . ThoughtCo . About.com .
</item 3>

<item 4>
For What It's Worth
`` What is Country Rock ? '' . ThoughtCo . About.com . Retrieved March 12 , 2017 . Jump up ^ Jim DeRogatis ( 1996 ) . Kaleidoscope Eyes : Psychedelic Rock from
the ' 60s to the ' 90s . Carol Publishing Group . p. 51 . ISBN 978 - 0 - 8065 - 1788 - 9 . Retrieved 13 October 2015 . Jump up ^ Google Books archive Billboard
March 25 , 1967 retrieved 25 March 2017 Jump up ^ `` Acclaimed Music Top 3000 songs '' . Acclaimed Music . 27 May 2009 . ^ Jump up to : David Browne ( November
11 , 2016 ) . `` ' For What It 's Worth ' : Inside Buffalo Springfield 's Classic Protest Song '' . Rolling Stone . Jump up ^ Gilliland , John ( 1969 ) . ``
Show 34 -- Revolt of the Fat Angel : American musicians respond to the British invaders .
</item 4>

<item 5>
Freaky Friday (song)
production Benny Blanco -- composition , production Twice as Nice -- composition , production Serban Ghenea -- mix engineering Patrizio Pigliarono --
engeineering Charts ( edit ) Chart ( 2018 ) Peak position Australia ( ARIA ) Austria ( Ö3 Austria Top 40 ) 20 Belgium ( Ultratop 50 Flanders ) 46 Belgium (
Ultratip Wallonia ) Canada ( Canadian Hot 100 ) 10 Denmark ( Tracklisten ) 13 France ( SNEP ) 113 Germany ( Official German Charts ) 21 Greece ( IFPI Greece )
16 Hungary ( Stream Top 40 ) 24 Ireland ( IRMA ) Netherlands ( Single Top 100 ) 28 New Zealand ( Recorded Music NZ ) Norway ( VG - lista ) 22 Portugal ( AFP )
28 Scotland ( Official Charts Company ) Sweden ( Sverigetopplistan ) 21 Switzerland ( Schweizer Hitparade ) 33 UK Singles ( Official Charts Company ) US
Billboard Hot 100 8 US Hot R&B / Hip - Hop Songs ( Billboard ) 5 US Rhythmic ( Billboard ) 16 Certifications ( edit ) Region Certification Certified units /
Sales Australia ( ARIA ) Gold 35,
</item 5>

<item 6>
For What It's Worth
Entertainment.time.com . Retrieved 2016 - 09 - 30 . Jump up ^ `` Oui 3 '' . Blair Booth Music . Retrieved 6 April 2016 . Jump up ^ `` Oui 3 '' . Official Charts
. Retrieved 6 April 2016 . Jump up ^ `` EXCLUSIVE ! Haley Reinhart Covering The ' 60s Classic For What It 's Worth Is As Chilling As Our Current Political
Climate ! Listen ! '' . PerezHilton . August 10 , 2017 . Retrieved August 10 , 2017 . External links ( edit ) Reasontorock analysis of song Lyrics of this song
at MetroLyrics Buffalo Springfield Richie Furay Dewey Martin Jim Messina Stephen Stills Neil Young Bruce Palmer Studio albums Buffalo Springfield Buffalo
Springfield Again Last Time Around Compilations and box sets Retrospective : The Best of Buffalo Springfield Buffalo Springfield ( compilation album ) Buffalo
Springfield ( box set ) Songs `` For What It 's Worth '' `` Sit Down , I Think I Love You '' `` Mr.
</item 6>

<item 7>
For What It's Worth
I Hope You Find It '' `` Take It Like a Man '' `` I Walk Alone '' Guest singles `` A Love Like Yours ( Do n't Come Knocking Everyday ) '' `` Dead Ringer for
Love '' `` I Got You Babe '' `` It Ai n't Necessarily So '' `` Love Can Build a Bridge '' `` Più che puoi '' `` Bewitched , Bothered and Bewildered '' Book
Portal Haley Reinhart Discography Videography Songs Studio albums Listen Up ! ( 2012 ) Better ( 2016 ) What 's That Sound ? ( 2017 ) EPs American Idol Season 10
Highlights : Haley Reinhart ( 2011 ) Songs `` Baby , It 's Cold Outside '' `` Free '' `` Ca n't Help Falling in Love '' `` Better '' Concert tours American
Idols LIVE !
</item 7>

<item 8>
For What It's Worth
( Part 2 ) : UNT Digital Library '' ( audio ) . Pop Chronicles . University of North Texas Libraries . ^ Jump up to : `` Sunset Strip Riots Closing of club
ignited the ' Sunset Strip riots ' '' . Los Angeles Times . 2007 - 08 - 05 . Retrieved 2012 - 01 - 16 . Jump up ^ Einarson , John ; Furay , Richie ( 2004 ) .
For What It 's Worth : The Story of Buffalo Springfield . Taylor Trade Publications . p. 127 . ISBN 978 - 0 - 8154 - 1281 - 6 . Jump up ^ Jimmy McDonough ( 2003
) . Shakey : Neil Young 's Biography . Random House . p. 201 . Jump up ^ `` Top 20 Pop Spotlight Cher - For What It 's Worth '' . Billboard . Nielsen Business
Media , Inc. 16 Aug 1969 . Jump up ^ Lustig , Jay ( 18 February 2011 ) .
</item 8>

<item 9>
For What It's Worth
Against the background of these civil disturbances , Stills recorded the song on December 5 , 1966 . Production ( edit ) Stills said in an interview that the
name of the song came about when he presented it to the record company executive Ahmet Ertegun who signed Buffalo Springfield to the Atlantic Records - owned
ATCO label . He said : `` I have this song here , for what it 's worth , if you want it . '' Another producer , Charlie Greene , claims that Stills first said
the above sentence to him , but credits Ahmet Ertegun with subtitling the single `` Stop , Hey What 's That Sound '' so that the song would be more easily
recognized . The song was recorded on December 5 , 1966 at Columbia Studios , Hollywood . Tom Dowd claims he mixed the song at Atlantic 's studio in New York ,
though this is disputed by others . He did take part in the production of Cher 's version of the song in 1969. . The song is distinguished with the use of the
guitar harmonics technique and sound . Legacy ( edit ) The song quickly became a well - known protest song .
</item 9>

<item 10>
For What It's Worth
Released January 1967 Format 7 - inch single Recorded December 5 , 1966 Studio Columbia Studios , Hollywood Genre Folk rock country rock psychedelic rock Length
2 : 37 Label Atco Songwriter ( s ) Stephen Stills Producer ( s ) Charles Greene Brian Stone Buffalo Springfield singles chronology `` Burned '' ( 1966 ) `` For
What It 's Worth '' ( 1967 ) `` Bluebird '' ( 1967 ) `` Burned '' ( 1966 ) `` For What It 's Worth '' ( 1967 ) `` Bluebird '' ( 1967 ) `` For What It 's Worth (
Stop , Hey What 's That Sound ) '' ( often referred to as simply `` For What It 's Worth '' ) is a song written by Stephen Stills . It was performed by Buffalo
Springfield , recorded on December 5 , 1966 , and released as a single on Atco Records in January 1967 . The single peaked at number seven on the Billboard Hot
100 chart .
</item 10>
=== END ITEM id=fe15c7c9d2988f8d ===

=== ITEM id=6995a98b38b98236 ===
Gold sentences and facts:
Sentence 0: Principal photography started April 26 , 1965 , at the 20th Century - Fox Studios and 20th Century - Fox Ranch , California . Other filming locations , simulating the desert , were Buttercup Valley , Arizona and Pilot Knob Mesa , California . The flying sequences were all filmed at Pilot Knob Mesa near Winterhaven , located in California 's Imperial Valley , on the western fringes of Yuma , Arizona .
  Fact 0.0: Principal photography started on April 26, 1965.
  Fact 0.1: Principal photography started at the 20th Century-Fox Studios in California.
  Fact 0.2: Principal photography started at the 20th Century-Fox Ranch in California.
  Fact 0.3: Other filming locations, which simulated the desert, were Buttercup Valley in Arizona.
  Fact 0.4: Other filming locations, which simulated the desert, were Pilot Knob Mesa in California.
  Fact 0.5: The flying sequences were all filmed at Pilot Knob Mesa.
  Fact 0.6: Pilot Knob Mesa is near Winterhaven.
  Fact 0.7: Pilot Knob Mesa is located in California's Imperial Valley.
  Fact 0.8: Pilot Knob Mesa is on the western fringes of Yuma, Arizona.

Retrieved items:
<item 1>
The Flight of the Phoenix (1965 film)
Bosley Crowther of the New York Times dismissed it as `` grim and implausible '' , while Variety praised the film as an `` often - fascinating and superlative
piece of filmmaking highlighted by standout performances and touches that show producer - director at his best . '' Awards ( edit ) The Flight of the Phoenix
was nominated for two Academy Awards : Ian Bannen for Supporting Actor and Michael Luciano for Film Editing . See also ( edit ) Flight of the Phoenix ( 2004
film ) , a re-make of this film Lady Be Good ( aircraft ) List of American films of 1965 List of film accidents Survival film , about the film genre , with a
list of related films Tragedy at Kufra References ( edit ) ^ Jump up to : Solomon , Aubrey ( 2002 ) . Twentieth Century Fox : A Corporate and Financial History
( 1st ed . ) . Lanham , Maryland : Scarecrow . ISBN 9780810842441 . Jump up ^ Silver , Alain ; Ursini , James ( 1995 ) . Whatever Happened to Robert Aldrich ?
</item 1>

<item 2>
The Flight of the Phoenix (1965 film)
Secaucus , New Jersey : Citadel Press , 1988 . ISBN 0 - 8065 - 1081 - 1 . External links ( edit ) The Flight of the Phoenix ( 1965 ) on IMDb The Flight of the
Phoenix at Rotten Tomatoes The Flight of the Phoenix at AllMovie The Jimmy Stewart Museum Home Page The Flight of the Phoenix at the TCM Movie Database The
Flight of the Phoenix at the American Film Institute Catalog ( hide ) Films directed by Robert Aldrich Big Leaguer ( 1953 ) World for Ransom ( 1954 ) Apache (
1954 ) Vera Cruz ( 1954 ) Kiss Me Deadly ( 1955 ) The Big Knife ( 1955 ) Autumn Leaves ( 1956 ) Attack ( 1956 ) Ten Seconds to Hell ( 1959 ) The Angry Hills (
1959 ) The Last Sunset ( 1961 ) Sodom and Gomorrah ( 1962 ) What Ever Happened to Baby Jane ? ( 1962 ) 4 for Texas ( 1963 ) Hush ... Hush ,
</item 2>

<item 3>
The Flight of the Phoenix (1965 film)
Although principal photography was completed on August 13 , 1965 , in order to complete filming , a North American O - 47 A ( N4725V ) from the Planes of Fame
Air Museum in Chino , California was modified and used as a flying Phoenix stand - in . With the canopy removed , a set of skids attached to the main landing
gear as well as ventral fin added to the tail , it sufficed as more - or-less a visual lookalike . Filming using the O - 47A was completed in October / November
1965 . It appears in the last flying scenes , painted to look like the earlier Phoenix P - 1 . The final production utilized a mix of footage that included the
O - 47A , the `` cobbled - together '' Phoenix and Phoenix P - 1 . The final credit on the screen was , `` It should be remembered ... that Paul Mantz , a fine
man and a brilliant flyer gave his life in the making of this film ... '' Reception ( edit ) The film opened in selected theaters on December 15 , 1965 , with a
full release in 1966 .
</item 3>

<item 4>
The Flight of the Phoenix (1965 film)
The flying sequences were all filmed at Pilot Knob Mesa near Winterhaven , located in California 's Imperial Valley , on the western fringes of Yuma , Arizona .
Aircraft used ( edit ) The Phoenix `` static '' model as seen in the film In 2005 , Hollywood aviation historian Simon Beck identified the aircraft used in the
film : Fairchild C - 82A Packet , N6887C -- flying shots . Fairchild C - 82A Packet , N4833V -- outdoor location wreck . Fairchild C - 82A Packet , N53228 --
indoor studio wreck . Fairchild R4Q - 1 Flying Boxcar ( the USMC C - 119C variant ) , BuNo. 126580 -- non-flying Phoenix prop . Tallmantz Phoenix P - 1 , N93082
-- flying Phoenix aircraft . North American O - 47 A , N4725V -- second flying Phoenix . The C - 82As were from Steward - Davies Inc. at Long Beach , California
, while the O - 47A came from the Planes of Fame air museum in California .
</item 4>

<item 5>
The Flight of the Phoenix (1965 film)
( January 2017 ) ( Learn how and when to remove this template message ) The Flight of the Phoenix 1965 theatrical release poster Directed by Robert Aldrich
Produced by Robert Aldrich Written by Lukas Heller Based on The Flight of the Phoenix by Elleston Trevor Starring James Stewart Richard Attenborough Hardy
Krüger Peter Finch Ernest Borgnine Ian Bannen Music by Frank De Vol Cinematography Joseph Biroc Edited by Michael Luciano Production company The Associates and
Aldrich Distributed by 20th Century Fox Release date December 15 , 1965 ( 1965 - 12 - 15 ) Running time 142 minutes Country United States Language English
Budget $5,355,000 or $3.8 million Box office $3 million ( US / Canada rentals ) 311,136 admissions ( France ) The Flight of the Phoenix is a 1965 American drama
film starring James Stewart , produced and directed by Robert Aldrich , and based on the 1964 novel The Flight of the Phoenix by Elleston Trevor .
</item 5>

<item 6>
The Flight of the Phoenix (1965 film)
Cast ( edit ) James Stewart as Capt . Frank Towns Richard Attenborough as Lew Moran Hardy Krüger as Heinrich Dorfmann Peter Finch as Capt . Harris Ernest
Borgnine as Trucker Cobb Ian Bannen as `` Ratbags '' Crow Ronald Fraser as Sgt . Watson Christian Marquand as Dr. Renaud Dan Duryea as Standish George Kennedy
as Mike Bellamy Gabriele Tinti as Gabriel Alex Montoya as Carlos Peter Bravos as Tasso William Aldrich as Bill Barrie Chase as Farida Stanley Ralph Ross as Arab
singer ( voice ; uncredited ) Production ( edit ) This section needs additional citations for verification . Please help improve this article by adding
citations to reliable sources . Unsourced material may be challenged and removed . ( January 2017 ) ( Learn how and when to remove this template message )
Locations ( edit ) Principal photography started April 26 , 1965 , at the 20th Century - Fox Studios and 20th Century - Fox Ranch , California . Other filming
locations , simulating the desert , were Buttercup Valley , Arizona and Pilot Knob Mesa , California .
</item 6>

<item 7>
The Flight of the Phoenix (1965 film)
at 08 : 30 . Text is available under the Creative Commons Attribution - ShareAlike License ; additional terms may apply . By using this site , you agree to the
Terms of Use and Privacy Policy . Wikipedia ® is a registered trademark of the Wikimedia Foundation , Inc. , a non-profit organization . About Wikipedia
</item 7>

<item 8>
The Flight of the Phoenix (1965 film)
org/w/index.php?title=The_Flight_of_the_Phoenix_(1965_film)&oldid=832173551 '' Categories : English - language films 20th Century Fox films 1965 films 1960s
disaster films 1960s drama films American adventure drama films American aviation films American disaster films American films Films scored by Frank De Vol
Films about aviation accidents or incidents Films based on British novels Films directed by Robert Aldrich Films set in deserts Films set in Libya Films set on
airplanes Films shot in Arizona Films shot in California American survival films Hidden categories : Wikipedia spam cleanup from January 2017 Wikipedia further
reading cleanup Pages using div col with deprecated parameters Articles needing additional references from January 2017 All articles needing additional
references Talk Contents About Wikipedia Català Deutsch Español Euskara Français Italiano Lëtzebuergesch Bahasa Melayu Nederlands 日本 語 Norsk Plattdüütsch Polski
Português Русский Srpskohrvatski / српскохрватски Svenska 8 more Edit links This page was last edited on 24 March 2018 ,
</item 8>

<item 9>
The Flight of the Phoenix (1965 film)
Jump up ^ Variety staff ( 1964 ) . Flight of the Phoenix , film review , Variety , December 31 , 1964 . Retrieved 2018 - 01 - 19 . Jump up ^ The Academy of
Motion Picture Arts and Sciences . `` The 38th Academy Awards , 1966 '' , honoring the films of 1965 ; awards presented at Santa Monica Civic Auditorium , Santa
Monica , California , April 18 , 1966 . Retrieved 2018 - 01 - 19 . Further reading ( edit ) Cox , Stephen . It 's a Wonderful Life : A Memory Book . Nashville ,
Tennessee : Cumberland House , 2003 . ISBN 1 - 58182 - 337 - 1 . Eliot , Mark . Jimmy Stewart : A Biography . New York : Random House , 2006 . ISBN 1 - 4000 -
5221 - 1 . Hardwick , Jack and Ed Schnepf . `` A Viewer 's Guide to Aviation Movies . '' The Making of the Great Aviation Films .
</item 9>

<item 10>
The Flight of the Phoenix (1965 film)
The Flight of the Phoenix ( 1965 film ) - wikipedia The Flight of the Phoenix ( 1965 film ) Jump to : navigation , search This article 's further reading may
not follow Wikipedia 's content policies or guidelines . Please improve this article by removing less relevant or redundant publications with the same point of
view ; or by incorporating the relevant publications into the body of the article through appropriate citations .
</item 10>
=== END ITEM id=6995a98b38b98236 ===

=== ITEM id=859b186c5b63e832 ===
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
The third pilot , which was made as an episode of Dick Powell 's Zane Grey Theater , was picked up by ABC and the first season of The Rifleman began filming in
July 1958 . Crawford was nominated for an Emmy Award , at age 13 , for his role as Mark McCain , the son of Lucas McCain , played by Chuck Connors , in the Four
Star Television series The Rifleman , which originally aired from 1958 to 1963 . Throughout The Rifleman 's five seasons , a remarkable on - screen chemistry
existed between Connors and Crawford in the depiction of their father - son relationship . They were still close friends when Connors died on November 10 , 1992
, and Crawford gave a eulogy at Connors ' memorial service . During the late 1950s and early 1960s , Crawford had wide popularity with American teenagers and a
recording career that generated four Billboard Top 40 hits , including the single , `` Cindy 's Birthday '' , which peaked at number 8 , in 1962 .
</item 1>

<item 2>
Rio Bravo (film)
Both remember how good with a gun Dude used to be . Chance 's friend Pat Wheeler ( Ward Bond ) and his wagon train of supplies stop in town , with a young
gunslinger , Colorado Ryan ( Ricky Nelson ) , riding guard . Inside the jail , Stumpy ( Walter Brennan ) , Chance 's game - legged deputy , keeps watch over the
jail and Joe , who knows that Stumpy holds an old grudge against Joe 's wealthy and powerful brother . Joe warns his jailers that Nathan Burdette will not like
how his brother is being treated . A mysterious woman nicknamed Feathers ( Angie Dickinson ) is in the saloon , playing poker . In the meantime , Dude and
Chance patrol the town . Hotel owner Carlos ( Pedro Gonzalez - Gonzalez ) stops the sheriff , saying Wheeler has been talking too much about Chance needing help
. In the saloon , Chance implores Wheeler to stop , as it will draw attention from the wrong people . Wheeler suggests that Colorado could be of assistance ,
but Colorado politely declines , saying he wants to `` mind my own business '' . Colorado also promises not to start any trouble without telling the sheriff
first .
</item 2>

<item 3>
Rio Bravo (film)
In this film , Robert Mitchum played the Dean Martin role , Arthur Hunnicutt the Walter Brennan character , and James Caan the Ricky Nelson role . Hawks again
named the Nelson / Caan character after a state ( in this case , Mississippi ) and in a wry , humorous twist on the original film , Hawks made him inept with
firearms , but skilled with a knife . The second remake , Rio Lobo , was made in 1970 with a plot much further off the original mold , starting with the absence
of a lawman - turned - drunkard character . This began with a Confederate train robbery of a Union gold shipment during the American Civil War , then moved to a
postwar Texas county thoroughly controlled by a rich , arrogant rancher . The heroes , with the exception of an old man similar to Brennan and Hunnicutt 's
characters in the previous pictures ( Jack Elam here ) , were complete outsiders . Along with Wayne and Elam , this movie starred Mexican film star Jorge Rivero
( as Frenchie ) , Christopher Mitchum ( Robert Mitchum 's son ) , and Jennifer O'Neill .
</item 3>

<item 4>
Johnny Crawford
Johnny Crawford - wikipedia Johnny Crawford Jump to : navigation , search For the Canadian ice hockey player and coach , see Johnny Crawford ( ice hockey ) .
Johnny Crawford Johnny Crawford in The Rifleman ( 1961 ) John Ernest Crawford ( 1946 - 03 - 26 ) March 26 , 1946 ( age 72 ) Los Angeles , California , U.S.
Occupation Actor , singer , musician , band leader Years active 1954 -- 96 Spouse ( s ) Charlotte Crawford ( 1995 -- present ) ( separated ) Relatives Robert L.
Crawford , Jr. ( brother ) Website Johnny Crawford fansite Johnny Crawford and Chuck Connors in The Rifleman ( 1960 ) John Ernest Crawford ( born March 26 ,
1946 ) is an American character actor , singer , and musician . At age 12 , Crawford rose to fame for playing Mark McCain , the son of Lucas McCain ( played by
Chuck Connors ) , in the popular ABC Western series , The Rifleman , which originally aired from 1958 to 1963 . Crawford first performed before a national
audience as a Mouseketeer .
</item 4>

<item 5>
Rio Bravo (film)
Because the film starred a crooner , Martin , and a teen idol , Nelson , Hawks included three songs in the soundtrack . Before the big showdown , in the jail
house , Martin sings `` My Rifle , My Pony , and Me '' ( which contained new lyrics to a Tiomkin tune that appeared in Red River ) accompanied by Nelson , after
which Nelson sings a brief version of `` Get Along Home , Cindy '' , accompanied by Martin and Brennan . Over the closing credits , Martin , backed by the
Nelson Riddle Orchestra , sings a specially composed song , `` Rio Bravo '' , written by Tiomkin with lyrics by Paul Francis Webster . Nelson later paid homage
to both the film and his character , Colorado , by including the song `` Restless Kid '' on his 1959 LP , Ricky Sings Again . Members of the Western Writers of
America chose `` My Rifle , My Pony , and Me '' as one of the Top 100 Western songs of all time . A brief clip from Rio Bravo was among the archive footage
later incorporated into the opening sequence of Wayne 's last film , The Shootist , to illustrate the backstory of Wayne 's character .
</item 5>

<item 6>
Johnny Crawford
His first important break as an actor followed with the title role in a Lux Video Theatre production of `` Little Boy Lost '' , a live NBC broadcast on March 15
, 1956 . He also appeared in the popular Western series The Lone Ranger , in 1956 , in one of the few color episodes of that series . Following that performance
, the young actor worked steadily with many seasoned actors and directors . Freelancing for 2 1 / 2 years , he accumulated almost 60 television credits ,
including featured roles in three episodes of NBC 's The Loretta Young Show and an appearance as Manuel in , `` I Am an American '' , an episode of the
syndicated crime drama Sheriff of Cochise . By the spring of 1958 , he had also performed 14 demanding roles in live teleplays for NBC 's Matinee Theatre ,
appeared on CBS 's sitcom , Mr. Adams and Eve , in the Wagon Train episode `` The Sally Potter Story '' ( in which Martin Milner also appeared ) and on the
syndicated series , Crossroads , Sheriff of Cochise , and Whirlybirds , and made three pilots of TV series .
</item 6>

<item 7>
Johnny Crawford
Jump up ^ `` Johnny Crawford '' . Scott Stander . Retrieved January 25 , 2014 . Jump up ^ `` Angry Woman -- Victoria Jackson on `` Tonight Show '' b / w `` .
Youtube . Retrieved January 25 , 2014 . Biography portal California portal Los Angeles portal Film portal Television portal Music portal United States Army
portal External links ( edit ) Johnny Crawford on The Rifleman Broncho Billy short film Johnny Crawford 's MySpace site Johnny Crawford on IMDb VIAF : 63364407
LCCN : no91003539 ISNI : 0000 0001 1526 4402 GND : 142142425 SUDOC : 079205798 BNF : cb11219552n ( data ) MusicBrainz : bf40b9eb - 068a - 4bdd - be46 -
ec0b5bd7a177 BNE : XX4832002 SNAC : w6sk5xbc Retrieved from `` https://en.wikipedia.org/w/index.
</item 7>

<item 8>
Johnny Crawford
He played a young deputy Billy Norris , in The Big Valley episode `` The Other Face Of Justice '' in 1969 . While enlisted in the United States Army for two
years , Crawford worked on training films as a production coordinator , assistant director , script supervisor , and occasional actor . His rank was sergeant at
the time of his honorable discharge in December 1967 . In 1968 , Crawford played a soldier wanted for murder in `` By the Numbers '' , an episode of the popular
TV series Hawaii Five - O . The Resurrection of Broncho Billy was a USC student film Crawford agreed to do as a favor to his close friend , producer John
Longenecker . It won the 1970 Academy Award for Best Live Action Short Subject . The Naked Ape was a partially animated 1973 feature film starring Crawford and
Victoria Principal , produced by Hugh Hefner . In 1976 , Johnny Crawford co-starred as Ben Shelby in the tenth episode of season three of Little House on the
Prairie , `` The Hunters '' . In 1985 , Crawford appeared as Noah Paisley , sheriff 's deputy , in an episode of Murder , She Wrote .
</item 8>

<item 9>
Homestead Acts
Wheeler Outlaws Billy the Kid Black Bart `` Curly Bill '' Brocius Butch Cassidy Billy Clanton Ike Clanton Bill Dalton Bill Doolin John Wesley Hardin Johnny
Ringo Jesse James Tom Ketchum Frank McLaury Tom McLaury Joaquin Murrieta Cochise County Cowboys Belle Starr Soapy Smith Sundance Kid Cole Younger Soldiers and
scouts Frederick Russell Burnham Kit Carson `` Buffalo Bill '' Cody Texas Jack Omohundro James C. Cooney George Crook George Armstrong Custer Samuel P.
Heintzelman Tom Horn Calamity Jane Luther Kelly Ranald S. Mackenzie Charley Reynolds Philip Sheridan Al Sieber Others John Jacob Astor William H. Boring
Jonathan R. Davis George Flavel C.S. Fly John Joel Glanton George E. Goodfellow Doc Holliday Seth Kinman Nat Love Sylvester Mowry Emperor Norton Annie Oakley
Thomas William Sweeny Jack Swilling Native Americans Apache Arapaho Arikara Assiniboine ( Nakota ) Blackfoot Cahuilla Cayuse Cheyenne Chinook Chippewa ( Ojibwe
) Caddo Cocopah Comanche Crow Dakota Five Civilized Tribes Hidatsa
</item 9>

<item 10>
Rio Bravo (film)
After a pair of sheer tights comes floating from a window to the street , a delighted Stumpy retrieves it , but Dude cautions him to mind his own business .
Cast ( edit ) John Wayne and Angie Dickinson in Rio Bravo John Wayne as John T. Chance Dean Martin as Dude Ricky Nelson as Colorado / Ryan Angie Dickinson as
Feathers Walter Brennan as Stumpy Ward Bond as Pat Wheeler John Russell as Nathan Burdette Pedro Gonzalez Gonzalez as Carlos Robante Estelita Rodriguez as
Consuela Robante Claude Akins as Joe Burdette Joseph Shimada as Burt Malcolm Atterbury and Harry Carey Jr. also receive a screen credit in the film 's opening ,
but their scenes were deleted from the final film . Production ( edit ) Ricky Nelson performing the song `` Get Along Home , Cindy '' in Rio Bravo Exteriors for
the film were shot at Old Tucson Studios , just outside Tucson . Filming took place in the summer of 1958 , and the movie 's credits gave 1958 as the year of
production , although the film was not released until 1959 .
</item 10>
=== END ITEM id=859b186c5b63e832 ===

=== ITEM id=e645f990b5059106 ===
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
Regina is brought into the chaos by trying to stop her sister and they find the Black Fairy and Gideon in the crystal mines , with the Black Fairy tricking
Zelena into utilizing her unstable Dark Magic to strengthen her for the Final Battle . Realizing that there is nothing left for her back in Oz , Zelena decides
to help the residents fight back against the Black Fairy by using the Crimson Heart to give up her magic , thereby undoing the mess she made in the mines . Emma
then uses her magic and one of the crystals to bring Mother Superior out of her coma with some help from Regina and Zelena . 130 19 `` The Black Fairy '' Alrick
Riley Jerome Schwartz & Dana Horgan April 30 , 2017 ( 2017 - 04 - 30 ) 3.05 The origin of Rumplestiltskin 's true destiny is revealed , as his mother Fiona
learns from the Blue Fairy and Tiger Lily that he was destined to become The Savior . When Fiona fears for her son and chooses power over protecting Rumple , it
results in her becoming the Black Fairy , and the Blue Fairy banishing her to the Dark Realm .
</item 1>

<item 2>
Once Upon a Time (season 6)
132 133 21 22 `` The Final Battle '' Steve Pearlman Ralph Hemecker Edward Kitsis & Adam Horowitz May 14 , 2017 ( 2017 - 05 - 14 ) 2.95 Fiona 's curse alters
Storybrooke , institutionalizing Emma and leaving only Henry and Gold `` awake , '' while sending Emma 's other allies to their home realms . The Final Battle
begins over Emma 's belief ; if she stops believing in the realms of story , they will be consumed by the Darkness and cease to exist . If all the realms are
destroyed , Fiona would gain unchecked powers and be able to break the Laws of Magic at will . In the Enchanted Forest , the banished characters gather at
Regina 's castle ; David and Hook obtain a magic bean and the Evil Queen sacrifices herself , holding off the Darkness to delay the group 's destruction . Fiona
convinces Emma to destroy the book , but Henry hand - draws one that restores her belief in time to save everyone in the castle .
</item 2>

<item 3>
Once Upon a Time (season 6)
He asks the Queen to kill Zelena , as the connection between his and Zelena 's hearts protects her from him . 120 9 `` Changelings '' Mairzee Almas David H.
Goodman & Brian Ridings November 27 , 2016 ( 2016 - 11 - 27 ) 3.28 In the Enchanted Forest , while served by Belle , Rumplestiltskin abducts a baby to use as
bait for the Black Fairy , the mother who abandoned him ; she escapes without answering his questions . In the present , Belle receives advice from her unborn
child while Gold becomes increasingly unhinged . Resenting Gold 's order to attack Zelena , the Evil Queen accelerates Belle 's pregnancy ; Mother Superior
delivers Belle 's son , Gideon , and sends him into hiding ; Gold vows to find him . As the Genie has been freed , Aladdin takes his place . Zelena and Regina
continue to feud . Emma obtains the sword from her vision .
</item 3>

<item 4>
Once Upon a Time (season 6)
In 1993 , a pixie flower awakens Snow and Charming from their cursed selves , but an awakened Rumplestiltskin tells them they must make a difficult decision to
ensure that Emma fulfills her destiny as the Savior . 129 18 `` Where Bluebirds Fly '' Michael Schultz David H. Goodman & Brigitte Hales April 23 , 2017 ( 2017
- 04 - 23 ) 2.69 In Zelena 's past , her childhood friend Stanum asks for her help in finding the Crimson Heart , a magical artifact that would be able to
restore his human form after having been cursed by the Wicked Witch of the North to slowly turn into tin . Zelena 's refusal to give up her magic leads to her
betraying Stanum , as she keeps the Heart and leaves him to become the Tin Man . In the Present , The Black Fairy tempts Zelena with an offer to join her , only
to have Zelena oppose her instead .
</item 4>

<item 5>
Once Upon a Time (season 6)
Zelena and Regina continue to feud . Emma obtains the sword from her vision . 121 10 `` Wish You Were Here '' Ron Underwood Edward Kitsis & Adam Horowitz
December 4 , 2016 ( 2016 - 12 - 04 ) 3.27 The Black Fairy abducts Gideon . Emma discovers the sword can kill the Evil Queen without hurting Regina . The Queen
steals the lamp and wishes Emma to an alternate reality where she was never the Savior ; Regina follows her . David and his allies take the lamp from the Queen
, who is turned into a serpent by the hooded figure , revealed to be the adult Gideon . Jasmine wishes herself and Aladdin to Agrabah . In the `` wish realm ''
, Emma lives as a princess until Regina traumatically restores her memory . Regina frees Rumplestiltskin in exchange for a magic bean , but the portal to
Storybrooke closes while Regina is distracted by the appearance of Robin Hood .
</item 5>

<item 6>
Once Upon a Time (season 6)
Cast and characters ( edit ) Main article : List of Once Upon a Time characters Regular ( edit ) Ginnifer Goodwin as Snow White / Mary Margaret Blanchard
Jennifer Morrison as Emma Swan Lana Parrilla as Evil Queen / Regina Mills Josh Dallas as Prince Charming / David Nolan Emilie de Ravin as Belle / Lacey Colin
O'Donoghue as Captain Hook / Killian Jones Jared S. Gilmore as Henry Mills Rebecca Mader as Wicked Witch of the West / Zelena Robert Carlyle as Rumplestiltskin
/ Mr. Gold Recurring ( edit ) Beverley Elliott as Widow Lucas / Granny Raphael Sbarge as Jiminy Cricket / Dr. Archie Hopper Lee Arenberg as Dreamy / Grumpy /
Leroy David Paul Grove as Doc Gabe Khouth as Sneezy / Mr. Clark Faustino Di Bauda as Sleepy / Walter Mig Macario as Bashful Geoff Gustafson as Stealthy Tony
Amendola as Geppetto / Marco Keegan Connor Tracy as the Blue Fairy / Mother Superior Eion Bailey as Pinocchio / August Booth David Anders as Victor Frankenstein
/ Dr.
</item 6>

<item 7>
Once Upon a Time (season 6)
Zelena moves in with Regina but , after learning her sister blames her for Robin Hood 's death , moves out and is approached by the Evil Queen . 113 `` A Bitter
Draught '' Ron Underwood Andrew Chambliss & Dana Horgan October 2 , 2016 ( 2016 - 10 - 02 ) 3.72 In the past in the Enchanted Forest , the Evil Queen offers to
aid the revenge plot of Edmond Dantes , The Count of Monte Cristo , if he kills Snow and David . When he falls in love with Snow 's handmaiden , he reneges and
flees with her to the Land of Untold Stories to save her from poison . In Storybrooke , the Evil Queen recruits Zelena and uses Edmond 's heart to set him
against Snow and David ; Regina kills Edmond to save the others , and realizes she retains the capacity for darkness . The Evil Queen declares that the
undesirable `` untold stories '' will be played out . She gives David evidence that his father 's death was not accidental .
</item 7>

<item 8>
Once Upon a Time (season 6)
Gold kills Fiona , who had hidden Belle and turned her into a coward ; this breaks the Curse , which restores everyone 's memories and reunites them in
Storybrooke . However , Gideon is still under orders to kill Emma . Gold attempts to free Gideon 's heart ; however , a spell cast by the Black Fairy prevents
him from doing so . Emma lays down her sword , allowing Gideon to impale her ; this releases Emma 's light magic and ends the battle . With both Gold and Emma
having chosen good , Gideon is returned to infancy and the realms are restored , and those who died in the crisis are revived . The characters embark on their
`` happy beginnings . '' Many years later in the New Enchanted Forest , an adult Henry vanishes while defending his daughter Lucy from an unidentified entity .
Guided by Tiger Lily , Lucy travels to Seattle , where Henry no longer remembers her .
</item 8>

<item 9>
Once Upon a Time (season 6)
Contents ( hide ) 1 Premise 2 Cast and characters 2.1 Regular 2.2 Recurring 2.3 Guest 3 Episodes 4 Production 4.1 Development 4.1. 1 Musical episode 4.2 Casting
5 Ratings 6 References 7 External links Premise ( edit ) The residents of Storybrooke are faced with threats from the Evil Queen and Mr. Hyde , both with
different agendas following the arrival of refugees from the Land of Untold Stories . This leads to events to test Emma 's savior abilities that results in the
arrival of the Black Fairy to begin the Final Battle that was prophesied centuries ago . As dark and light collides , a new , and darker curse is unleashed ,
leading to the fall of all the realms with magic . The Final Battle is fought and won , leading to the heroes earning their happy beginnings , but a new
adventure begins in the far future for an adult Henry involving his daughter Lucy , and another curse looming across the New Enchanted Forest inhabitants .
</item 9>

<item 10>
Once Upon a Time (season 6)
Emma and Gideon finally meet for the battle , but Emma survives , choosing her own fate . Gideon retreats , proclaiming this far from over . Gold and Belle
decide to work together for the good of their son . 123 12 `` Murder Most Foul '' Morgan Beggs Jerome Schwartz & Jane Espenson March 12 , 2017 ( 2017 - 03 - 12
) 3.06 The events of how Robert , the father of David and James , was murdered are detailed in a flashback , in which Robert learns what happened to James after
he ran away from King George . With help from Rumple , he tracks him to Pleasure Island , but their reunion is cut short by King George , who orders Robert 's
execution . His death affects David in the present day after he sees visions of Robert and asks Hook to help him , despite his being reformed . When Hook
becomes concerned that David is taking this investigation too far by going after Albert , he intervenes and brings David back to sanity . Hook is also worried
about whether David will give him his blessing to marry Emma , but David does .
</item 10>
=== END ITEM id=e645f990b5059106 ===

=== ITEM id=194402643480a4db ===
Gold sentences and facts:
Sentence 0: The story of Shakespeare 's Hamlet was derived from the legend of Amleth , preserved by 13th - century chronicler Saxo Grammaticus in his Gesta Danorum , as subsequently retold by the 16th - century scholar François de Belleforest . Shakespeare may also have drawn on an earlier Elizabethan play known today as the Ur - Hamlet , though some scholars believe he himself wrote the Ur - Hamlet , later revising it to create the version of Hamlet we now have . He almost certainly wrote his version of the title role for his fellow actor , Richard Burbage , the leading tragedian of Shakespeare 's time . In the 400 years since its inception , the role has been performed by numerous highly acclaimed actors in each successive century .
  Fact 0.0: The story of Shakespeare's Hamlet was derived from the legend of Amleth.
  Fact 0.1: The legend of Amleth was preserved by the 13th-century chronicler Saxo Grammaticus in his Gesta Danorum.
  Fact 0.2: The legend of Amleth was subsequently retold by the 16th-century scholar François de Belleforest.
  Fact 0.3: Shakespeare may also have drawn on an earlier Elizabethan play known today as the Ur-Hamlet.
  Fact 0.4: Some scholars believe Shakespeare himself wrote the Ur-Hamlet and later revised it to create the version of Hamlet we have today.
  Fact 0.5: Shakespeare almost certainly wrote his version of the title role for his fellow actor Richard Burbage.
  Fact 0.6: Richard Burbage was the leading tragedian of Shakespeare's time.
  Fact 0.7: In the 400 years since the role of Hamlet was created, it has been performed by numerous highly acclaimed actors.
  Fact 0.8: The role of Hamlet has been performed by highly acclaimed actors in each successive century for 400 years.

Retrieved items:
<item 1>
Hamlet
( father ) Mary Arden ( mother ) Gilbert Shakespeare ( brother ) Joan Shakespeare ( sister ) Edmund Shakespeare ( brother ) Richard Shakespeare ( grandfather )
John Hall ( son - in - law ) Thomas Quiney ( son - in - law ) Thomas Nash ( grandson - in - law ) * Shakespeare and other authors Lost William Shakespeare 's
Hamlet Characters Hamlet Claudius Gertrude Ghost Polonius Laertes Ophelia Horatio Rosencrantz and Guildenstern Fortinbras The Gravediggers Yorick Soliloquies ``
To be , or not to be '' `` Mortal coil '' `` What a piece of work is a man '' `` Speak the speech '' Words and phrases `` The lady doth protest too much ,
methinks '' `` Thy name is '' `` Primrose path '' Terminology Dumbshow Induction Quiddity Substitution Sources Criticism Legend of Hamlet The Spanish Tragedy Ur
- Hamlet Critical approaches Bibliographies Horwendill Saxo Grammaticus House of Gonzaga Damon and Pythias Influence Common phrases from Hamlet Hamlet in
popular culture References to Ophelia Language of flowers Human skull symbolism Performances Moscow Art Theatre ( 1911 -- 1912 )
</item 1>

<item 2>
Hamlet
Jr. , in the American Company 's production in Philadelphia in 1759 . David Garrick 's iconic hand gesture expresses Hamlet 's shock at the first sight of the
ghost ( artist : unknown ) . John Philip Kemble made his Drury Lane debut as Hamlet in 1783 . His performance was said to be 20 minutes longer than anyone else
's , and his lengthy pauses provoked the suggestion by Richard Brinsley Sheridan that `` music should be played between the words '' . Sarah Siddons was the
first actress known to play Hamlet ; many women have since played him as a breeches role , to great acclaim . In 1748 , Alexander Sumarokov wrote a Russian
adaptation that focused on Prince Hamlet as the embodiment of an opposition to Claudius 's tyranny -- a treatment that would recur in Eastern European versions
into the 20th century . In the years following America 's independence , Thomas Apthorpe Cooper , the young nation 's leading tragedian , performed Hamlet among
other plays at the Chestnut Street Theatre in Philadelphia , and at the Park Theatre in New York .
</item 2>

<item 3>
Hamlet
Firm evidence for specific early performances of the play is scant . What is known is that the crew of the ship Red Dragon , anchored off Sierra Leone ,
performed Hamlet in September 1607 ; that the play toured in Germany within five years of Shakespeare 's death ; and that it was performed before James I in
1619 and Charles I in 1637 . Oxford editor George Hibbard argues that , since the contemporary literature contains many allusions and references to Hamlet (
only Falstaff is mentioned more , from Shakespeare ) , the play was surely performed with a frequency that the historical record misses . All theatres were
closed down by the Puritan government during the Interregnum . Even during this time , however , playlets known as drolls were often performed illegally ,
including one called The Grave - Makers based on Act 5 , Scene 1 of Hamlet . Restoration and 18th century ( edit ) Title page and frontispiece for Hamlet ,
Prince of Denmark : A Tragedy . As it is now acted at the Theatres - Royal in Drury - Lane and Covent - Garden . London , 1776 The play was revived early in the
Restoration .
</item 3>

<item 4>
Hamlet
London , 1776 The play was revived early in the Restoration . When the existing stock of pre-civil war plays was divided between the two newly created patent
theatre companies , Hamlet was the only Shakespearean favourite that Sir William Davenant 's Duke 's Company secured . It became the first of Shakespeare 's
plays to be presented with movable flats painted with generic scenery behind the proscenium arch of Lincoln 's Inn Fields Theatre . This new stage convention
highlighted the frequency with which Shakespeare shifts dramatic location , encouraging the recurrent criticism of his failure to maintain unity of place . In
the title role , Davenant cast Thomas Betterton , who continued to play the Dane until he was 74 . David Garrick at Drury Lane produced a version that adapted
Shakespeare heavily ; he declared : `` I had sworn I would not leave the stage till I had rescued that noble play from all the rubbish of the fifth act . I have
brought it forth without the grave - digger 's trick , Osrick , & the fencing match '' . The first actor known to have played Hamlet in North America is Lewis
Hallam . Jr.
</item 4>

<item 5>
Hamlet
Hamlet - wikipedia Hamlet Jump to : navigation , search This article is about the play by William Shakespeare . For other uses , see Hamlet ( disambiguation ) .
The American actor Edwin Booth as Hamlet , ca . 1870 The Tragedy of Hamlet , Prince of Denmark , often shortened to Hamlet ( / ˈhæmlɪt / ) , is a tragedy
written by William Shakespeare at an uncertain date between 1599 and 1602 . Set in Denmark , the play dramatises the revenge Prince Hamlet is called to wreak
upon his uncle , Claudius , by the ghost of Hamlet 's father , King Hamlet . Claudius had murdered his own brother and seized the throne , also marrying his
deceased brother 's widow . Hamlet is Shakespeare 's longest play , and is considered among the most powerful and influential works of world literature , with a
story capable of `` seemingly endless retelling and adaptation by others '' .
</item 5>

<item 6>
Hamlet
It was probably one of Shakespeare 's most popular works during his lifetime , and still ranks among his most performed , topping the performance list of the
Royal Shakespeare Company and its predecessors in Stratford - upon - Avon since 1879 . It has inspired many other writers -- from Johann Wolfgang von Goethe and
Charles Dickens to James Joyce and Iris Murdoch -- and has been described as `` the world 's most filmed story after Cinderella '' . The story of Shakespeare 's
Hamlet was derived from the legend of Amleth , preserved by 13th - century chronicler Saxo Grammaticus in his Gesta Danorum , as subsequently retold by the 16th
- century scholar François de Belleforest . Shakespeare may also have drawn on an earlier Elizabethan play known today as the Ur - Hamlet , though some scholars
believe he himself wrote the Ur - Hamlet , later revising it to create the version of Hamlet we now have . He almost certainly wrote his version of the title
role for his fellow actor , Richard Burbage , the leading tragedian of Shakespeare 's time .
</item 6>

<item 7>
Hamlet
In stark contrast to earlier opulence , William Poel 's 1881 production of the Q1 text was an early attempt at reconstructing the Elizabethan theatre 's
austerity ; his only backdrop was a set of red curtains . Sarah Bernhardt played the prince in her popular 1899 London production . In contrast to the ``
effeminate '' view of the central character that usually accompanied a female casting , she described her character as `` manly and resolute , but nonetheless
thoughtful ... ( he ) thinks before he acts , a trait indicative of great strength and great spiritual power '' . In France , Charles Kemble initiated an
enthusiasm for Shakespeare ; and leading members of the Romantic movement such as Victor Hugo and Alexandre Dumas saw his 1827 Paris performance of Hamlet ,
particularly admiring the madness of Harriet Smithson 's Ophelia . In Germany , Hamlet had become so assimilated by the mid-19th century that Ferdinand
Freiligrath declared that `` Germany is Hamlet '' . From the 1850s , the Parsi theatre tradition in India transformed Hamlet into folk performances , with
dozens of songs added .
</item 7>

<item 8>
Hamlet
Jenkins suggests that any personal satire may be found in the name `` Polonius '' , which might point to a Polish or Polonian connection . G.R. Hibbard
hypothesised that differences in names ( Corambis / Polonius : Montano / Raynoldo ) between the First Quarto and other editions might reflect a desire not to
offend scholars at Oxford University . Date ( edit ) John Barrymore as Hamlet ( 1922 ) `` Any dating of Hamlet must be tentative '' , cautions the New Cambridge
editor , Phillip Edwards . The earliest date estimate relies on Hamlet 's frequent allusions to Shakespeare 's Julius Caesar , itself dated to mid-1599 . The
latest date estimate is based on an entry , of 26 July 1602 , in the Register of the Stationers ' Company , indicating that Hamlet was `` latelie Acted by the
Lo : Chamberleyne his servantes '' . In 1598 , Francis Meres published his Palladis Tamia , a survey of English literature from Chaucer to its present day ,
within which twelve of Shakespeare 's plays are named .
</item 8>

<item 9>
Hamlet
He is dethroned by the spectre of an actor , and we shall never be able to keep the usurper out of our dreams . Maurice Maeterlinck in La Jeune Belgique ( 1890
) . Shakespeare 's day to the Interregnum ( edit ) Shakespeare almost certainly wrote the role of Hamlet for Richard Burbage . He was the chief tragedian of the
Lord Chamberlain 's Men , with a capacious memory for lines and a wide emotional range . Judging by the number of reprints , Hamlet appears to have been
Shakespeare 's fourth most popular play during his lifetime -- only Henry IV Part 1 , Richard III and Pericles eclipsed it . Shakespeare provides no clear
indication of when his play is set ; however , as Elizabethan actors performed at the Globe in contemporary dress on minimal sets , this would not have affected
the staging . Firm evidence for specific early performances of the play is scant .
</item 9>

<item 10>
Hamlet
In the 400 years since its inception , the role has been performed by numerous highly acclaimed actors in each successive century . Three different early
versions of the play are extant : the First Quarto ( Q1 , 1603 ) ; the Second Quarto ( Q2 , 1604 ) ; and the First Folio ( F1 , 1623 ) . Each version includes
lines and entire scenes missing from the others . The play 's structure and depth of characterisation have inspired much critical scrutiny . One such example is
the centuries - old debate about Hamlet 's hesitation to kill his uncle , which some see as merely a plot device to prolong the action , but which others argue
is a dramatisation of the complex philosophical and ethical issues that surround cold - blooded murder , calculated revenge , and thwarted desire . More
recently , psychoanalytic critics have examined Hamlet 's unconscious desires , while feminist critics have re-evaluated and attempted to rehabilitate the often
maligned characters of Ophelia and Gertrude . Contents ( hide ) 1 Characters 2 Plot 2.1 Act I 2.2 Act II 2.3 Act III 2.
</item 10>
=== END ITEM id=194402643480a4db ===

=== ITEM id=9610ca41b70701e3 ===
Gold sentences and facts:
Sentence 0: A UCC - 1 financing statement ( an abbreviation for Uniform Commercial Code - 1 ) is a legal form that a creditor files to give notice that it has or may have an interest in the personal property of a debtor ( a person who owes a debt to the creditor as typically specified in the agreement creating the debt ) . This form is filed in order to `` perfect '' a creditor 's security interest by giving public notice that there is a right to take possession of and sell certain assets for repayment of a specific debt with a certain priority . Such notices of sale are often found in the local newspapers . Once the form has been filed , the creditor establishes a relative priority with other creditors of the debtor . This process is also called `` perfecting the security interest '' in the property , and this type of loan is a secured loan . A financing statement may also be filed in the real estate records by a lessor of fixtures to establish the priority of the lessor 's rights against a holder of a mortgage or other lien on the real property . The creditor 's rights against the debtor and the lessor 's rights against the lessee are based on the credit documents and the lease , respectively , and not the financing statement .
  Fact 0.0: A UCC-1 financing statement is a legal form.
  Fact 0.1: UCC-1 is an abbreviation for Uniform Commercial Code-1.
  Fact 0.2: A creditor files a UCC-1 financing statement to give notice that it has or may have an interest in the personal property of a debtor.
  Fact 0.3: A debtor is a person who owes a debt to the creditor, as typically specified in the agreement creating the debt.
  Fact 0.4: A financing statement is filed to perfect a creditor's security interest.
  Fact 0.5: The filing gives public notice that there is a right to take possession of and sell certain assets for repayment of a specific debt, with a certain priority.
  Fact 0.6: Such notices of sale are often found in local newspapers.
  Fact 0.7: Once the form has been filed, the creditor establishes a relative priority with other creditors of the debtor.
  Fact 0.8: Perfecting the security interest is also called perfecting the security interest in the property.
  Fact 0.9: This type of loan is a secured loan.
  Fact 0.10: A financing statement may also be filed in the real estate records by a lessor of fixtures.
  Fact 0.11: The real estate filing establishes the lessor's priority against a holder of a mortgage or other lien on the real property.
  Fact 0.12: The creditor's rights against the debtor are based on the credit documents, not the financing statement.
  Fact 0.13: The lessor's rights against the lessee are based on the lease, not the financing statement.
Sentence 1: The financing statement is generally filed with the office of the state secretary of state , in the state where the debtor is located - for an individual , the state where the debtor resides , for most kinds of business organizations the state of incorporation or organization . Many states have a state agency that operates under the secretary of state , which is tasked with overseeing business organizations and activities , including receipt of financing statements . However , an exception exists if the collateral is something that is tied to a particular piece of real property , such as timber , mineral rights , or fixtures . In that case , the filing must be made in the county where the property is located , usually in the recording office or county court , because that is where third parties are most likely to search for such record .
  Fact 1.0: The financing statement is generally filed with the office of the state secretary of state.
  Fact 1.1: The financing statement is generally filed in the state where the debtor is located.
  Fact 1.2: For an individual, the state where the debtor is located is the state where the debtor resides.
  Fact 1.3: For most kinds of business organizations, the state where the debtor is located is the state of incorporation or organization.
  Fact 1.4: Many states have a state agency that operates under the secretary of state.
  Fact 1.5: That state agency is tasked with overseeing business organizations and activities, including receipt of financing statements.
  Fact 1.6: An exception exists if the collateral is tied to a particular piece of real property, such as timber, mineral rights, or fixtures.
  Fact 1.7: In that exception case, the filing must be made in the county where the property is located.
  Fact 1.8: In that exception case, the filing is usually made in the recording office or county court.
  Fact 1.9: Third parties are most likely to search for such a record in that county.

Retrieved items:
<item 1>
IRS penalties
Jump up ^ 26 USC 6651 ( a ) ( 1 ) . Jump up ^ IRC § 6651 ( a ) ( 1 ) Jump up ^ IRC § 6651 ( a ) ( 2 ) Jump up ^ `` 1040 ( 2015 ) '' . Internal Revenue Service .
Retrieved June 28 , 2016 . Jump up ^ 26 USC 6662 . Jump up ^ 26 USC 6721 . Jump up ^ 26 CFR 301 - 6721 - 1 ( a ) ( 2 ) ( ii ) . Jump up ^ 26 USC 6698 . Jump up
^ 26 USC 6699 . Jump up ^ Technically , the Form 1120S is an income tax return , not an information return . Jump up ^ 26 USC 3101 - 3406 . Jump up ^ 26 USC
6672 . Jump up ^ 26 USC 3504 . Jump up ^ See , e.g. , United States v. St. Clair , 45 Amer . Fed . Tax Rep.
</item 1>

<item 2>
UCC-1 financing statement
UCC - 1 financing statement - wikipedia UCC - 1 financing statement Jump to : navigation , search This article needs additional citations for verification .
Please help improve this article by adding citations to reliable sources . Unsourced material may be challenged and removed . ( October 2014 ) ( Learn how and
when to remove this template message ) A UCC - 1 financing statement ( an abbreviation for Uniform Commercial Code - 1 ) is a legal form that a creditor files
to give notice that it has or may have an interest in the personal property of a debtor ( a person who owes a debt to the creditor as typically specified in the
agreement creating the debt ) . This form is filed in order to `` perfect '' a creditor 's security interest by giving public notice that there is a right to
take possession of and sell certain assets for repayment of a specific debt with a certain priority . Such notices of sale are often found in the local
newspapers . Once the form has been filed , the creditor establishes a relative priority with other creditors of the debtor . This process is also called ``
perfecting the security interest '' in the property , and this type of loan is a secured loan .
</item 2>

<item 3>
A UCC-1 publicly records a creditor’s claimed security interest in personal property to help establish priority; it lists debtor, creditor, and collateral, with
most filings made statewide and some land-related filings made locally. Bonds are tradable debt securities with specified principal, maturity, coupons, and
covenants; prices and yields move inversely. They vary by issuer, payout, currency, market, and seniority, and carry interest-rate, credit, inflation, and
liquidity risks. Examples include climate
</item 3>

<item 4>
United Kingdom corporation tax
Individuals could set off the tax credit against their income tax liability . On introduction , ACT was set at 30 % of the gross dividend ( the actual amount
paid plus the tax credit ) . If a company made a £ 70 dividend payment to an individual , the company would pay £ 30 of advance corporation tax . The
shareholder would receive the £ 70 cash payment , plus a tax credit of £ 30 ; thus , the individual would be deemed to have earned £ 100 , and to have already
paid tax of £ 30 on it . The ACT paid by the company would be deductible against its final `` mainstream '' corporation tax bill . To the extent that the
individual 's tax on the dividend was less than the tax credit -- for example , if his income was too low to pay tax ( below £ 595 in 1973 -- 1974 ) -- he would
be able to reclaim some or all of the £ 30 tax paid by the company .
</item 4>

<item 5>
UCC-1 financing statement
References ( edit ) Jump up ^ http://www.findlegalforms.com/articles/form-encyclopedia/what-is-a-ucc1-financing-statement Jump up ^
http://biztaxlaw.about.com/od/glossaryu/g/ucc1statement.htm Retrieved from ``
https://en.wikipedia.org/w/index.php?title=UCC-1_financing_statement&oldid=803697764 '' Categories : Uniform Commercial Code Legal terminology Hidden categories
: Articles needing additional references from October 2014 All articles needing additional references Talk About Wikipedia Add links This page was last edited
on 4 October 2017 , at 02 : 00 . About Wikipedia
</item 5>

<item 6>
UCC-1 financing statement
A financing statement may also be filed in the real estate records by a lessor of fixtures to establish the priority of the lessor 's rights against a holder of
a mortgage or other lien on the real property . The creditor 's rights against the debtor and the lessor 's rights against the lessee are based on the credit
documents and the lease , respectively , and not the financing statement . Pursuant to the standards set forth in the UCC , at 9 - 503 and 9 - 504 , the
financing statement need only contain three pieces of information : the debtor 's name and address the creditor 's name and address an indication of the
collateral , `` whether or not it is specific , if it reasonably identifies what is described . '' ( UCC 9 - 108 ) The financing statement is generally filed
with the office of the state secretary of state , in the state where the debtor is located - for an individual , the state where the debtor resides , for most
kinds of business organizations the state of incorporation or organization . Many states have a state agency that operates under the secretary of state , which
is tasked with overseeing business organizations and activities , including receipt of financing statements .
</item 6>

<item 7>
A UCC-1 financing statement is a public notice filed by a creditor to disclose a claimed security interest in a debtor’s personal property and establish
priority among creditors, helping “perfect” a secured interest. It generally lists the debtor’s name and address, creditor’s name and address, and collateral
description. Filings usually go to the relevant state office; those involving fixtures or certain property tied to land are filed in the county where the
property is located. The statement itself does not create a lien or
</item 7>

<item 8>
IRS penalties
U.S. corporations more than 25 % owned , directly or indirectly , by foreign persons must file Form 5472 to report such ownership and all transactions with
related parties . Failure to timely file carries a $10,000 penalty per required form . This penalty may be increased by $10,000 per month per form for continued
failure to file . In addition , taxpayers who fail to report changes in foreign taxes used as credits against Federal income tax may be subject to penalties .
U.S. citizen or resident taxpayers ( including entities ) who are beneficiaries of a foreign trust or make transfers of property to a foreign trust must report
information about the transfer and the trust or corporation . Failure to timely report on Form 3520 or Form 3520 - A may result in penalties of up to 35 % .
Similar transferors to foreign corporations failing to file Form 926 may face penalties of 10 % of the value of the transfer , up to $100,000 . Excise taxes as
penalties ( edit ) Federal excise taxes are imposed on a variety of goods and services . Some of these taxes require purchase of tax stamps or other evidence of
advance payment of tax .
</item 8>

<item 9>
UCC-1 financing statement
However , an exception exists if the collateral is something that is tied to a particular piece of real property , such as timber , mineral rights , or fixtures
. In that case , the filing must be made in the county where the property is located , usually in the recording office or county court , because that is where
third parties are most likely to search for such record . In the case of a loan secured by personal property collateral , the filing of a financing statement
gives notice of a lien against the property so that other lenders or buyers of the personal property will know of the security interest . In the case of a
filing of a financing statement by a lessor of fixtures , the filing of the financing statement gives notice of the lessor 's interests to others who acquire an
interest in the real property and related fixtures . The financing statement does not create a lien nor does it create any additional rights against a lessee in
favor of a lessor , the filing of a financing statement just gives notice of whatever rights the creditor or lessor have under their loan documents or lease ,
respectively .
</item 9>

<item 10>
Ownership
A structured group duly constituted as an entity under law may still not protect members from being personally liable for each other 's actions . Court
decisions against the entity itself may give rise to unlimited personal liability for each and every member . An example of this situation is a professional
partnership ( e.g. law practice ) in some jurisdictions . Thus , being a partner or owner in a group may give little advantage in terms of share ownership while
producing a lot of risk to the partner , owner or participant . Sharing gains ( edit ) At the end of each financial year , accounting rules determine a surplus
or profit , which may be retained inside the entity or distributed among owners according to the initial setup intent when the entity was created . For public
corporations , common shareholders have no right to receive any of the profit . Entities with a member focus will give financial surplus back to members
according to the volume of financial activity that the participating member generated for the entity . Examples of this are producer cooperatives , buyer
cooperatives and participating whole life policyholders in both mutual and share - capital insurance companies . Entities with share voting rights that depend
on financial capital distribute surplus among shareholders without regard to any other contribution to the entity .
</item 10>
=== END ITEM id=9610ca41b70701e3 ===
