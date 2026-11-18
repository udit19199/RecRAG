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

=== ITEM id=e1314441bbd4f80a ===
Gold sentences and facts:
Sentence 0: The Proud Family is a soundtrack album for the show of the same name . A combination of both original tunes by the characters in the show , popular hits by popular music artists such as Alicia Keys , Jhené Aiko , India Arie , and Solange Knowles & Destiny 's Child ( who perform the theme song ) , and classic soul music from artists such as Aretha Franklin and The O'Jays .
  Fact 0.0: The Proud Family is a soundtrack album.
  Fact 0.1: The Proud Family soundtrack album is for the show of the same name.
  Fact 0.2: The album combines original tunes by the characters in the show.
  Fact 0.3: The album combines popular hits by popular music artists such as Alicia Keys, Jhené Aiko, India Arie, and Solange Knowles & Destiny's Child.
  Fact 0.4: Solange Knowles & Destiny's Child perform the theme song.
  Fact 0.5: The album combines classic soul music from artists such as Aretha Franklin and The O'Jays.

Retrieved items:
<item 1>
The Partridge Family
The Partridge Family - wikipedia The Partridge Family Jump to : navigation , search The Partridge Family Genre Musical sitcom Created by Bernard Slade Starring
Shirley Jones David Cassidy Danny Bonaduce Susan Dey Suzanne Crough Jeremy Gelbwaks Dave Madden Brian Forster Theme music composer Diane Hilderbrand Danny
Janssen Wes Farrell Opening theme `` When We 're Singin ' '' ( 1970 -- 71 ) `` C'mon , Get Happy '' ( 1971 -- 74 ) Composer ( s ) George Duning Benny Golson
Warren Barker Hugo Montenegro Shorty Rogers Country of origin United States Original language ( s ) English No. of seasons No. of episodes 96 ( list of episodes
) Production Executive producer ( s ) Bob Claver Producer ( s ) William Bickley Paul Junger Witt Dale McRaven Larry Rosen Mel Swope Cinematography Fred Jackman
, Jr .
</item 1>

<item 2>
You'll Never Find Another Love Like Mine
in which Lou Rawls himself sings it to Damon Wayans during a colonoscopy , That ' 70s Show ( 1998 -- 2006 ) , South Park ( 1997 -- ) , Two and a Half Men ( 2003
-- 2015 ) , and Psych ( 2006 -- 2014 ) . The song could also be heard in an episode of The Proud Family , where Lou Rawls sings and dances with Penny Proud . In
addition , it was performed by the singer in doll form in an Action League Now ! segment on KaBlam ! . The song `` If You Could Love Me '' by Edwyn Collins uses
many of the same chords , and has a similar sound . Rocco DiSpirito and Karina Smirnoff also danced the rumba to this song in week 2 of season 7 of Dancing with
the Stars .
</item 2>

<item 3>
NBC Sunday Night Football
Singer Pink sang the theme song for the broadcast in 2006 , a reworking of the Joan Jett song `` I Hate Myself for Loving You '' retitled `` Waiting All Day for
Sunday Night '' . Several alternative versions were used throughout the season , substituting different lyrics when appropriate ( such as `` Waiting All Year
For Opening Night '' ) . In 2007 , country singer Faith Hill replaced Pink as the singer of the opening theme , and a new arrangement of the Joan Jett song
coincided with her debut . The Faith Hill intro , in particular , was not without criticism and parody . The intro was lampooned in the October 9 , 2010 episode
of Saturday Night Live , with host Jane Lynch as Hill ( with Jason Sudeikis as Al Michaels and Bill Hader as Cris Collinsworth ) . In the 30 Rock episode ``
Season 4 '' , the character of Jenna Maroney ( Jane Krakowski ) sings what appears to be an allusion to the Faith Hill intro for NBC 's fictional Tennis Night
in America program .
</item 3>

<item 4>
The Proud Family (soundtrack)
The Proud Family ( soundtrack ) - Wikipedia The Proud Family ( soundtrack ) Jump to : navigation , search This article does not cite any sources . Please help
improve this article by adding citations to reliable sources . Unsourced material may be challenged and removed . ( December 2009 ) ( Learn how and when to
remove this template message ) The Proud Family Soundtrack album by Various Artists Released April 20 , 2004 August 16 , 2005 ( Re-release ) Genre Soul , hip
hop , hip hop soul Label Walt Disney The Proud Family is a soundtrack album for the show of the same name . A combination of both original tunes by the
characters in the show , popular hits by popular music artists such as Alicia Keys , Jhené Aiko , India Arie , and Solange Knowles & Destiny 's Child ( who
perform the theme song ) , and classic soul music from artists such as Aretha Franklin and The O'Jays . References ( edit ) This soundtrack - related article is
a stub . You can help Wikipedia by expanding it .
</item 4>

<item 5>
The Partridge Family
More than one song was used as the theme song . Season 1 episodes feature the song `` When We 're Singin ' '' , which was replaced in subsequent seasons with ``
C'mon Get Happy '' . `` Five of us , and Mom working all day , we knew we could help her if our music would pay , Danny got Reuben to sell our song , and it
really came together when Mom sang along ... '' ( from `` When We 're Singin ' '' ) Later , under Farrell 's direction , the new version had new lyrics sung to
the `` When We 're Singin ' '' tune . With the new chorus finalized , `` C'mon Get Happy '' showcased the new verse : `` We had a dream , we 'd go travelin '
together , We 'd spread a little lovin ' then we 'd keep movin ' on .
</item 5>

<item 6>
The Proud Family (soundtrack)
You can help Wikipedia by expanding it . Retrieved from `` https://en.wikipedia.org/w/index.php?title=The_Proud_Family_(soundtrack)&oldid=638462881 ''
Categories : Television animation soundtracks 2004 soundtracks Hip hop soundtracks Soul soundtracks Walt Disney Records soundtracks Soundtrack stubs Hidden
categories : Articles lacking sources from December 2009 All articles lacking sources Articles with hAudio microformats Music infoboxes with deprecated
parameters All stub articles Talk About Wikipedia Add links This page was last edited on 17 December 2014 , at 07 : 09 . Text is available under the Creative
Commons Attribution - ShareAlike License ; additional terms may apply . By using this site , you agree to the Terms of Use and Privacy Policy . Wikipedia ® is a
registered trademark of the Wikimedia Foundation , Inc. , a non-profit organization . About Wikipedia
</item 6>

<item 7>
The Partridge Family
Actor David Cassidy sang lead on most of the records , and actor Shirley Jones can be heard on a few . But the other voices and all the instruments were
recorded by session musicians . The Partridge Family 's instrumental sound was provided by musicians informally known as The Wrecking Crew , who did many of the
pop hits of the day attributed to other performers . The actors pretended to play their instruments . None of the actors sang background ; they lip - synched to
recorded singing by The Love Generation , AKA The Ron Hicklin Singers , composed of brothers John and Tom Bahler , Jackie Ward ( who as Robin Ward had a # 14
hit `` Wonderful Summer '' on Dot Records in 1963 ) , and Ron Hicklin . Actor David Cassidy was originally to lip sync with the rest of the cast , but convinced
Farrell that he could sing and was allowed to join the studio ensemble as the lead singer . He and his step - mother , Shirley , are the only two show actors
allowed to sing for recording . The Partridge Family theme song , produced by Rogers , was shown over opening credits . More than one song was used as the theme
song .
</item 7>

<item 8>
The Partridge Family
L-R : Shirley Jones , Jeremy Gelbwaks , Suzanne Crough , Susan Dey , Danny Bonaduce and David Cassidy In the pilot episode , a group of musical siblings in the
fictitious city of San Pueblo , California ( said to be `` 40 miles from Napa County '' in episode 24 , A Partridge By Any Other Name ) convinces their widowed
mother , bank teller Shirley Partridge , to help them out by singing as they record a pop song in their garage . Through the efforts of precocious 10 - year -
old Danny they find a manager , Reuben Kincaid , who helps make the song a Top 40 hit . After more persuading , Shirley agrees that the family can go on tour .
They acquire an old school bus , a 1957 Chevrolet Series 6800 Superior , for touring , paint it with Mondrian - inspired patterns , and head to Las Vegas ,
Nevada for their first live gig at Caesars Palace . Subsequent episodes usually feature the band performing in various venues or in their garage . The shows
often contrast suburban life with the adventures of a show - business family on the road .
</item 8>

<item 9>
NBC Sunday Night Football
Houston Chronicle . Retrieved September 18 , 2006 . Jump up ^ `` John Williams Sunday Night Football Theme -- Song -- MP3 Stream '' . IMEEM Music . Jump up ^ ``
Joel Beckerman Creates New Arrangement For This Year 's Superbowl ( sic ) '' . Man Made Music . SHM . Archived from the original on September 30 , 2011 .
Retrieved November 15 , 2011 . Jump up ^ Schlussel , Debbie ( August 2006 ) . `` From Hank to ... Pink ? : NFL Sunday Night Adds Unneeded Girl Power '' . Jump
up ^ `` ' Sunday Night Football ' Thinks Pink '' . Zap2it . Tribune Company . August 30 , 2006 . Retrieved September 19 , 2006 . Jump up ^ `` Multi-platinum pop
singer Pink performs `` NBC Sunday Night Football '' opening music `` . NBC . The Futon Critic . August 30 , 2006 . Retrieved November 1 , 2006 . Jump up ^ ``
Faith Hill to sing Sunday Night Football theme '' .
</item 9>

<item 10>
NBC Sunday Night Football
A new Underwood - sung theme known as `` Oh , Sunday Night '' , which takes elements from her 2014 duet hit with Miranda Lambert , `` Somethin ' Bad '' ,
premiered with the opening game of the 2016 season on September 11 . For NBC 's coverage of Thursday Night Football , a capella group Pentatonix sings the
opening theme song called `` Weekend Go . '' The song is a reworked version of their original song `` Sing . '' NBC also used the `` TNF '' main theme music
called `` Ca n't Hold Us Down , '' performed by members of the Broadway orchestra for Hamilton . Show opening ( edit ) The song is at the centerpiece of the
opening montage , which has changed several times over the years . Underwood 's music has always played in the background over the official welcome after the
opening is completed and the teams take the field . 2006 ( edit ) For the first season , Pink appeared to sing from the top of a skyscraper as a helicopter
zoomed down on a city skyline with enlarged players Shaun Alexander , LaDainian Tomlinson and Tom Brady and the field , the results of computer - generated
imagery .
</item 10>
=== END ITEM id=e1314441bbd4f80a ===

=== ITEM id=c8dcafd3f893bda7 ===
Gold sentences and facts:
Sentence 0: The Rolling Stones The Rolling Stones performing at Summerfest in Milwaukee in 2015 . Left to right : Charlie Watts , Ronnie Wood , Mick Jagger and Keith Richards Background information Origin London , England Genres Rock blues Years active 1962 -- present Labels Decca London Rolling Stones Virgin ABKCO Interscope Polydor Associated acts The New Barbarians Billy Preston Faces The Pretty Things John Mayall & the Bluesbreakers Bill Wyman 's Rhythm Kings Website rollingstones.com Members Mick Jagger Keith Richards Charlie Watts Ronnie Wood Past members Brian Jones Ian Stewart Dick Taylor Bill Wyman Mick Taylor
  Fact 0.0: The Rolling Stones performed at Summerfest in Milwaukee in 2015.
  Fact 0.1: The photo shows, from left to right, Charlie Watts, Ronnie Wood, Mick Jagger and Keith Richards.
  Fact 0.2: The Rolling Stones have origin London, England.
  Fact 0.3: The Rolling Stones' genres are rock and blues.
  Fact 0.4: The Rolling Stones have been active from 1962 to the present.
  Fact 0.5: The Rolling Stones' labels include Decca, London, Rolling Stones, Virgin, ABKCO, Interscope, and Polydor.
  Fact 0.6: The Rolling Stones' associated acts include The New Barbarians, Billy Preston, Faces, The Pretty Things, John Mayall & the Bluesbreakers, and Bill Wyman's Rhythm Kings.
  Fact 0.7: The Rolling Stones' website is rollingstones.com.
  Fact 0.8: The current members of the Rolling Stones are Mick Jagger, Keith Richards, Charlie Watts, and Ronnie Wood.
  Fact 0.9: The past members of the Rolling Stones include Brian Jones, Ian Stewart, Dick Taylor, Bill Wyman, and Mick Taylor.
Sentence 1: Mick Jagger -- vocals , harmonica , rhythm guitar , percussion , keyboards ( 1962 -- present ) Keith Richards -- rhythm and lead guitar , bass guitar , vocals ( 1962 -- present ) Charlie Watts -- drums , percussion ( 1963 -- present ) Ronnie Wood -- lead and rhythm guitar , bass guitar , backing vocals ( 1975 -- present )
  Fact 1.0: Mick Jagger plays vocals, harmonica, rhythm guitar, percussion, and keyboards, and has been active from 1962 to the present.
  Fact 1.1: Keith Richards plays rhythm and lead guitar, bass guitar, and vocals, and has been active from 1962 to the present.
  Fact 1.2: Charlie Watts plays drums and percussion, and has been active from 1963 to the present.
  Fact 1.3: Ronnie Wood plays lead and rhythm guitar, bass guitar, and backing vocals, and has been active from 1975 to the present.
Sentence 2: The Rolling Stones are an English rock band formed in London , England in 1962 . The first stable line - up consisted of Brian Jones ( guitar , harmonica ) , Mick Jagger ( lead vocals ) , Keith Richards ( guitar , backing vocals ) , Bill Wyman ( bass ) , Charlie Watts ( drums ) , and Ian Stewart ( piano ) . Stewart was removed from the official line - up in 1963 but continued as a touring member until his death in 1985 . Jones left the band less than a month before his death in 1969 , having already been replaced by Mick Taylor , who remained until 1974 . After Taylor left the band , Ronnie Wood took his place in 1975 and has been on guitar in tandem with Richards ever since . Following Wyman 's departure in 1993 , Darryl Jones joined as their touring bassist . Touring keyboardists for the band have been Nicky Hopkins ( 1967 -- 1982 ) , Ian McLagan ( 1978 -- 1981 ) , Billy Preston ( through the mid-1970s ) and Chuck Leavell ( 1982 -- present ) . The band was first led by Brian Jones , but after developing into the band 's songwriters , Jagger and Richards assumed leadership when Jones was dealing with personal troubles and legal issues .
  Fact 2.0: The Rolling Stones are an English rock band.
  Fact 2.1: The Rolling Stones were formed in London, England in 1962.
  Fact 2.2: The first stable line-up of the Rolling Stones consisted of Brian Jones, Mick Jagger, Keith Richards, Bill Wyman, Charlie Watts, and Ian Stewart.
  Fact 2.3: Brian Jones played guitar and harmonica in the first stable line-up.
  Fact 2.4: Mick Jagger was the lead vocalist in the first stable line-up.
  Fact 2.5: Keith Richards played guitar and backing vocals in the first stable line-up.
  Fact 2.6: Bill Wyman played bass in the first stable line-up.
  Fact 2.7: Charlie Watts played drums in the first stable line-up.
  Fact 2.8: Ian Stewart played piano in the first stable line-up.
  Fact 2.9: Ian Stewart was removed from the official line-up in 1963.
  Fact 2.10: Ian Stewart continued as a touring member of the Rolling Stones until his death in 1985.
  Fact 2.11: Brian Jones left the Rolling Stones less than a month before his death in 1969.
  Fact 2.12: Mick Taylor replaced Brian Jones in the band.
  Fact 2.13: Mick Taylor remained in the band until 1974.
  Fact 2.14: Ronnie Wood took Mick Taylor's place in 1975 after Taylor left the band.
  Fact 2.15: Ronnie Wood has been on guitar in tandem with Keith Richards ever since 1975.
  Fact 2.16: Darryl Jones joined the Rolling Stones as their touring bassist after Bill Wyman's departure in 1993.
  Fact 2.17: Bill Wyman departed from the Rolling Stones in 1993.
  Fact 2.18: Nicky Hopkins was a touring keyboardist for the Rolling Stones from 1967 to 1982.
  Fact 2.19: Ian McLagan was a touring keyboardist for the Rolling Stones from 1978 to 1981.
  Fact 2.20: Billy Preston was a touring keyboardist for the Rolling Stones through the mid-1970s.
  Fact 2.21: Chuck Leavell has been a touring keyboardist for the Rolling Stones since 1982.
  Fact 2.22: The band was first led by Brian Jones.
  Fact 2.23: Jagger and Richards became the band's songwriters.
  Fact 2.24: Jagger and Richards assumed leadership of the band when Brian Jones was dealing with personal troubles and legal issues.

Retrieved items:
<item 1>
The Rolling Stones
4 1968 -- 1972 : `` Back to basics '' 1.5 1972 -- 1977 : Critical fluctuations and Ronnie Wood 1.6 1978 -- 1982 : Commercial peak 1.7 1983 -- 1988 : Band
turmoil and solo efforts 1.8 1989 -- 1999 : Comeback , return to popularity , and record - breaking tours 1.9 2000 -- 2011 : A Bigger Bang and continued success
1.10 2012 -- present : 50th anniversary and covers album 2 Musical development 3 Legacy 4 Tours 5 Band members 6 Discography 7 See also 8 Notes 9 References 9.1
Footnotes 9.2 Sources 10 Further reading 11 External links History ( edit ) Early History ( edit ) Keith Richards and Mick Jagger became childhood friends and
classmates in 1950 in Dartford , Kent , before the Jagger family moved to Wilmington , five miles ( 8.05 km ) away , in 1954 .
</item 1>

<item 2>
The Rolling Stones
keyboards , piano , backing vocals ( 1962 -- 1993 ; guest 2012 ) Early members Dick Taylor -- bass guitar ( 1962 ) Ricky Fenson -- bass guitar ( 1962 -- 1963 )
Colin Golding -- bass guitar ( 1962 -- 1963 ) Tony Chapman -- drums ( 1962 -- 1963 ) Carlo Little -- drums ( 1962 -- 1963 ) Current touring members Chuck
Leavell -- keyboards , backing vocals ( 1982 -- present ) Bernard Fowler -- backing vocals , percussion ( 1989 -- present ) Darryl Jones -- bass guitar ,
backing vocals ( 1993 -- present ) Matt Clifford -- keyboards , French horn , musical integrator ( 1989 -- 1990 , 2012 -- present ) Tim Ries -- saxophone ,
keyboards ( 1999 -- present ) Karl Denson -- saxophone ( 2014 -- present ) Sasha Allen -- backing vocals , co-lead vocals on `` Gimme Shelter '' ( 2016 --
present ) Former touring musicians Bobby Keys -- saxophone ( 1969 -- 1973 ; 1981 -- 2014 ;
</item 2>

<item 3>
The Rolling Stones
The group sent a tape of their best recordings to Korner , who was favourably impressed . On 7 April , Korner visited Ealing Jazz Club , where they met the
members of Blues Incorporated , who included the slide guitarist Brian Jones , the keyboardist Ian Stewart and the drummer Charlie Watts . After a meeting with
Korner , Jagger and Richards started jamming with the group . Jones , no longer in a band , advertised for bandmates in Jazz Weekly , while Stewart found them a
practice space ; together they decided to start a band playing Chicago blues . Soon after , Jagger , Taylor and Richards left Blues Incorporated to join Jones
and Stewart . At the first rehearsal were also the guitarist Geoff Bradford and the vocalist Brian Knight , both of whom decided not to join the band , citing
objections to playing the Chuck Berry and Bo Diddley songs preferred by Jagger and Richards . In June 1962 the line - up of Jagger , Richards , Jones , Stewart
and Taylor was completed with the addition of the drummer Tony Chapman . According to Richards , Jones named the band during a phone call to Jazz News .
</item 3>

<item 4>
The Rolling Stones
The Rolling Stones - wikipedia The Rolling Stones Jump to : navigation , search `` Rolling Stones '' redirects here . For other uses , see Rolling Stones (
disambiguation ) . For the magazine , see Rolling Stone . The Rolling Stones The Rolling Stones performing at Summerfest in Milwaukee in 2015 . Left to right :
Charlie Watts , Ronnie Wood , Mick Jagger and Keith Richards Background information Origin London , England Genres Rock blues Years active 1962 -- present
Labels Decca London Rolling Stones Virgin ABKCO Interscope Polydor Associated acts The New Barbarians Billy Preston Faces The Pretty Things John Mayall & the
Bluesbreakers Bill Wyman 's Rhythm Kings Website rollingstones.com Members Mick Jagger Keith Richards Charlie Watts Ronnie Wood Past members Brian Jones Ian
Stewart Dick Taylor Bill Wyman Mick Taylor The Rolling Stones are an English rock band formed in London , England in 1962 . The first stable line - up consisted
of Brian Jones ( guitar , harmonica ) , Mick Jagger ( lead vocals ) , Keith Richards ( guitar , backing vocals ) , Bill Wyman ( bass ) , Charlie Watts ( drums )
, and Ian Stewart ( piano ) .
</item 4>

<item 5>
The Rolling Stones
Stewart was removed from the official line - up in 1963 but continued as a touring member until his death in 1985 . Jones left the band less than a month before
his death in 1969 , having already been replaced by Mick Taylor , who remained until 1974 . After Taylor left the band , Ronnie Wood took his place in 1975 and
has been on guitar in tandem with Richards ever since . Following Wyman 's departure in 1993 , Darryl Jones joined as their touring bassist . Touring
keyboardists for the band have been Nicky Hopkins ( 1967 -- 1982 ) , Ian McLagan ( 1978 -- 1981 ) , Billy Preston ( through the mid-1970s ) and Chuck Leavell (
1982 -- present ) . The band was first led by Brian Jones , but after developing into the band 's songwriters , Jagger and Richards assumed leadership when
Jones was dealing with personal troubles and legal issues .
</item 5>

<item 6>
The Rolling Stones
Gomelsky , who had no written agreement with the band , was not consulted . Initially , Oldham tried to apply the strategy used by Brian Epstein , the Beatles '
manager : making the band members wear suits . Later , he changed his mind and imagined a band which contrasted with the Beatles , featuring unmatched clothing
, long hair , an unclean appearance . He later said he wanted to make the Stones `` a raunchy , gamy , unpredictable bunch of undesirables '' and to ``
establish that the Stones were threatening , uncouth and animalistic '' . Stewart left the official line - up , but remained road manager and touring
keyboardist . On Stewart 's decision , Oldham later said `` Well , he just does n't look the part , and six is too many for ( fans ) to remember the faces in
the picture . '' Later , Oldham reduced the ages of the band members in publicity to make them appear as teenagers . Decca Records , which had declined to enter
into a deal with the Beatles , gave the Rolling Stones a recording contract with favourable terms .
</item 6>

<item 7>
The Rolling Stones
( 1972 ) Goats Head Soup ( 1973 ) It 's Only Rock ' n Roll ( 1974 ) Black and Blue ( 1976 ) Some Girls ( 1978 ) Emotional Rescue ( 1980 ) Tattoo You ( 1981 )
Undercover ( 1983 ) Dirty Work ( 1986 ) Steel Wheels ( 1989 ) Voodoo Lounge ( 1994 ) Bridges to Babylon ( 1997 ) A Bigger Bang ( 2005 ) Blue & Lonesome ( 2016 )
See also ( edit ) The Rolling Stones portal Book : The Rolling Stones The Rolling Stones Museum List of highest - grossing concert tours Notes ( edit ) Jump up
^ Mick Avory himself has categorically denied `` on many occasions '' that he played with the Rollin ' Stones that night . In fact he only rehearsed twice with
them in the Bricklayers Arms pub , before they became known as the Rollin ' Stones . Jump up ^ Wyman 's book Rolling With The Stones incorrectly states the band
played the Alcove club that night . Jump up ^ The comma in the title of the song was later dropped .
</item 7>

<item 8>
The Rolling Stones
bass guitar , vocals ( 1962 -- present ) Charlie Watts -- drums , percussion ( 1963 -- present ) Ronnie Wood -- lead and rhythm guitar , bass guitar , backing
vocals ( 1975 -- present ) Former members Brian Jones -- rhythm guitar , harmonica , keyboards , piano , backing vocals ( 1962 -- 1969 ; died 1969 ) Ian Stewart
-- keyboards , piano ( 1962 -- 1963 ; touring 1969 , 1975 -- 1976 , 1978 , 1981 -- 1982 ; died 1985 ) Mick Taylor -- lead guitar , bass guitar , backing vocals
( 1969 -- 1974 ; guest 1981 , 2012 -- 2014 ) Bill Wyman -- bass guitar , keyboards , piano , backing vocals ( 1962 -- 1993 ;
</item 8>

<item 9>
The Rolling Stones
According to Richards , Jones named the band during a phone call to Jazz News . When asked by a journalist for the band 's name , Jones saw a Muddy Waters LP
lying on the floor ; one of the tracks was `` Rollin ' Stone '' . 1962 -- 1964 : building a following ( edit ) The back room of what was the Crawdaddy Club in
Richmond , London where the Rolling Stones had their first residency in 1963 Jones , Jagger , Richards , Stewart and Taylor played a gig billed as `` the Rollin
' Stones '' on 12 July 1962 , at the Marquee Club in London . Shortly afterwards the band went on their first tour of the UK , which they called a `` training
ground '' tour , because it was a new experience for all of them . Their material included the Chicago blues as well as Chuck Berry and Bo Diddley songs . At
that time , the line - up included neither bassist Bill Wyman , who joined in December 1962 , nor drummer Charlie Watts , who joined in January 1963 , thus
completing the band 's original rhythm section .
</item 9>

<item 10>
The Rolling Stones
By 1963 they were finding their musical stride as well as popularity , and in 1964 two unscientific opinion polls rated them as Britain 's most popular group ,
even outranking the Beatles . The name of the band was changed shortly after their first gig to `` The Rolling Stones '' . The group 's then acting manager
Giorgio Gomelsky secured a Sunday afternoon residency at the Crawdaddy Club in Richmond in February 1963 , which , Gomelsky claimed , triggered an ``
international renaissance for the blues '' . In May 1963 , The Rolling Stones signed Andrew Loog Oldham as their manager ; he was a former publicist who had
been directed to the band by his previous clients , the Beatles . Because Oldham was only nineteen and had not reached the age of majority -- he was also
younger than anyone in the band -- he could not get an agent 's licence nor sign any contracts without his mother also signing . By necessity he joined with
booking agent Eric Easton in order to secure record financing and assistance booking venues . Gomelsky , who had no written agreement with the band , was not
consulted .
</item 10>
=== END ITEM id=c8dcafd3f893bda7 ===

=== ITEM id=04dd00d039bc0132 ===
Gold sentences and facts:
Sentence 0: The theory is generally associated with Hirschman . He presented a complete theoretical formulation of the strategy . Underdeveloped countries display common characteristics : low levels of GNI per capita and slow GNI per capita growth , large income inequalities and widespread poverty , low levels of productivity , great dependence on agriculture , a backward industrial structure , a high proportion of consumption and low savings , high rates of population growth and dependency burdens , high unemployment and underemployment , technological backwardness and dualism ( existence of both traditional and modern sectors ) . In a less - developed country , these characteristics lead to scarce resources or inadequate infrastructure to exploit these resources . With a lack of investors and entrepreneurs , cash flows can not be directed into various sectors that influence balanced economic growth .
  Fact 0.0: The theory is generally associated with Hirschman.
  Fact 0.1: Hirschman presented a complete theoretical formulation of the strategy.
  Fact 0.2: Underdeveloped countries display common characteristics.
  Fact 0.3: Underdeveloped countries have low levels of GNI per capita.
  Fact 0.4: Underdeveloped countries have slow GNI per capita growth.
  Fact 0.5: Underdeveloped countries have large income inequalities.
  Fact 0.6: Underdeveloped countries have widespread poverty.
  Fact 0.7: Underdeveloped countries have low levels of productivity.
  Fact 0.8: Underdeveloped countries have great dependence on agriculture.
  Fact 0.9: Underdeveloped countries have a backward industrial structure.
  Fact 0.10: Underdeveloped countries have a high proportion of consumption and low savings.
  Fact 0.11: Underdeveloped countries have high rates of population growth and dependency burdens.
  Fact 0.12: Underdeveloped countries have high unemployment and underemployment.
  Fact 0.13: Underdeveloped countries have technological backwardness.
  Fact 0.14: Underdeveloped countries have dualism, meaning both traditional and modern sectors exist.
  Fact 0.15: In a less-developed country, these characteristics lead to scarce resources or inadequate infrastructure to exploit these resources.
  Fact 0.16: In a less-developed country, a lack of investors and entrepreneurs means cash flows cannot be directed into various sectors that influence balanced economic growth.

Retrieved items:
<item 1>
Strategy of unbalanced growth
Convergence occurs as the output of external economies diminishes at each step . Growth sequences tend to move towards convergence or divergence and the policy
is usually concerned with preventing rapid convergence and promoting the possibility of divergence . External economies ( edit ) New projects often appropriate
external economies created by preceding ventures and create external economies that may be utilized by subsequent ones . Sometimes the project undertaken
creates external economies , causing private profit to fall short of what is socially desirable . The reverse is also possible . Some ventures have a larger
input of external economies than the output . Therefore , Hirschman says , `` the projects that fall into this category must be net beneficiaries of external
economies '' . Social Overhead capital ( edit ) Social Overhead Capital ( SOC ) is defined as basic services without which primary , secondary and tertiary
productive activities can not function . In a narrow sense , Social Overhead Capital is defined to include transportation and electricity , while in a wider
sense , it includes all public services , including law and order and education . Criteria for classifying an asset as Social Overhead Capital include : The
services provided by the activity should facilitate a great variety of economic activities . The services provided should be subject to public control .
</item 1>

<item 2>
Strategy of unbalanced growth
( hide ) Development Economics : Partial Theories of Development Dual - sector model Big push model Critical minimum effort theory Ragnar Nurkse 's balanced
growth theory Strategy of unbalanced growth Social dualism Technological dualism Fei -- Ranis model of economic growth Retrieved from ``
https://en.wikipedia.org/w/index.php?title=Strategy_of_unbalanced_growth&oldid=838124154 '' Categories : Economic growth Hidden categories : Wikipedia articles
that are incomprehensible from August 2016 All Wikipedia articles that are incomprehensible Articles lacking in - text citations from January 2012 All articles
lacking in - text citations All articles with unsourced statements Articles with unsourced statements from February 2013 Wikipedia articles needing
clarification from February 2013 Articles with unsourced statements from March 2012 All articles with specifically marked weasel - worded phrases Articles with
specifically marked weasel - worded phrases from February 2013 Talk Contents About Wikipedia Deutsch Français 中文 Edit links This page was last edited on 25
April 2018 , at 02 : 21 . Text is available under the Creative Commons Attribution - ShareAlike License ; additional terms may apply .
</item 2>

<item 3>
Strategy of unbalanced growth
The services provided should be subject to public control . The services can not be imported . The investment needed to provide services should be characterized
by some unevenness as well as by high capital output ratio . Development via capital imbalances ( edit ) The strategy of unbalanced growth has been discussed
within the frameworks of development through shortage of SOC and development through excess of SOC . In the first case , the country invests in direct
productive activities ( DPA ) . Direct productive activity increases demand for SOC , inducing investment . In the second case , SOC expands , which reduces the
cost of services , inducing investment in DPA . The cost of producing any unit of output of DPA is inversely proportional to SOC . The economy 's major
objective is to attain increasing output of DPA . One of the paradoxes of development is that poor countries can not afford to be economical . According to
Hirschman , resources are not scarce per se , but the ability to employ those resources may be lacking . To explain unbalanced growth , Hirschman assumes that
the country invests in either DPA or SOC .
</item 3>

<item 4>
Strategy of unbalanced growth
Paul Streeten , `` Unbalanced Growth . '' Economic Integration , Sythoff , Leiden ( Netherlands ) , 1961 . Reprinted in A.N. Agarwala and S.P. Singh , ( eds . )
, Accelerating Investment in Developing Economies ( London : Oxford University Press , 1969 ) .
</item 4>

<item 5>
Strategy of unbalanced growth
In a less - developed country , these characteristics lead to scarce resources or inadequate infrastructure to exploit these resources . With a lack of
investors and entrepreneurs , cash flows can not be directed into various sectors that influence balanced economic growth . Hirschman contends that deliberate
unbalancing of the economy according to the strategy is the best method of development and if the economy is to be kept moving ahead , the task of development
policy is to maintain tension , disproportions and disequilibrium . Balanced growth should not be the goal but rather the maintenance of existing imbalances ,
which can be seen from profit and losses . Therefore , the sequence that leads away from equilibrium is precisely an ideal pattern for development . Unequal
development of various sectors often generates conditions for rapid development . More - developed industries provide undeveloped industries an incentive to
grow . Hence , development of underdeveloped countries should be based on this strategy . The path of unbalanced growth is described by three phases :
Complementary Induced investment External economies Singer believed that desirable investment programs always exist within a country that represent unbalanced
investment to complement the existing imbalance . These investments create a new imbalance , requiring another balancing investment .
</item 5>

<item 6>
Strategy of unbalanced growth
Strategy of unbalanced growth - Wikipedia Strategy of unbalanced growth Jump to : navigation , search This article may be incomprehensible or very hard to
understand . In particular , it uses both very general statements which are not further clarified or contextualized , and technical jargon that is not explained
. Please help by rewording it if the intended meaning can be determined . The talk page may have details . ( August 2016 ) ( Learn how and when to remove this
template message ) This article includes a list of references , related reading or external links , but its sources remain unclear because it lacks inline
citations . Please help to improve this article by introducing more precise citations . ( January 2012 ) ( Learn how and when to remove this template message )
Unbalanced growth is a natural path of economic development . Situations that countries are in at any one point in time reflect their previous investment
decisions and development . Accordingly , at any point in time desirable investment programs that are not balanced investment packages may still advance welfare
. Unbalanced investment can complement or correct existing imbalances . Once such an investment is made , a new imbalance is likely to appear , requiring
further compensating investments .
</item 6>

<item 7>
Strategy of unbalanced growth
Therefore , growth need not take place in a balanced way . Supporters of the unbalanced growth doctrine include Albert O. Hirschman , Hans Singer , Paul
Streeten , Marcus Fleming , Prof. Rostov and J. Sheehan . Contents ( hide ) 1 Introduction 2 Complementarity 3 Induced investment 4 External economies 5 Social
Overhead Capital 6 Development via capital imbalances 7 Backward and forward linkages 8 Linkages and last industries 8.1 Disadvantages 9 Further reading 10
Notes 11 References Introduction ( edit ) The theory is generally associated with Hirschman . He presented a complete theoretical formulation of the strategy .
Underdeveloped countries display common characteristics : low levels of GNI per capita and slow GNI per capita growth , large income inequalities and widespread
poverty , low levels of productivity , great dependence on agriculture , a backward industrial structure , a high proportion of consumption and low savings ,
high rates of population growth and dependency burdens , high unemployment and underemployment , technological backwardness and dualism ( existence of both
traditional and modern sectors ) .
</item 7>

<item 8>
Strategy of unbalanced growth
These investments create a new imbalance , requiring another balancing investment . One sector will always grow faster than another , so the need for unbalanced
growth will continue as investments must complement existing imbalance . Hirschman states `` If the economy is to be kept moving ahead , the task of development
policy is to maintain tensions , disproportions and disequilibrium '' . This situation exists for all societies , developed or underdeveloped . Complementarity
( edit ) Complementarity is a situation where increased production of one good or service builds up demand for the second good or service . When the second
product is privately produced , this demand will lead to imports or higher domestic production of the second product , as it will be in the interests of the
producers to do so . Otherwise , the increased demand takes the form of political pressure . This is the case for such public services such as law and order ,
education , water and electricity that can not reasonably be imported . Induced investment ( edit ) Complementarity allows investment in one industry or sector
to encourage investment in others . This concept of induced investment is like a multiplier , because each investment triggers a series of subsequent events .
</item 8>

<item 9>
Strategy of unbalanced growth
11 , No. 2 ( Jun. , 1959 ) , pp. 167 -- 190 Notes ( edit ) References ( edit ) Hirschman , Albert O . `` The Strategy of Economic Development , '' in Agarwal ,
A.N. and Singh , S.P. ( eds ) , Accelerating Investment in Developing Economies ( London Oxford Press , 1969 ) Thirwall , A.P. Growth and Development : With
special reference to developing economies ( London : Macmillian Press ltd ... , sixth edition . 1999 ) , pp237 -- 242 . Scitovsky , Tibor . Papers on welfare
and Growth ( London George Allen and Unwin ltd ... 1965 ) Refer to 5th paper entitled `` Growth -- Balanced and Unbalanced '' . Nath , S.K. , `` Balanced Growth
. '' in Livingstone ( ed ) , Economic Policy for Development ( Harmodworth , Middlesex : Penguin Books ltd. 1971 ) Han Singer , `` The Concept of Balanced
Growth and Economic Development ; Theory and Facts , '' University of Texas Conference on Economic Development , April 1958 .
</item 9>

<item 10>
Graph query row:
person: Albert O. Hirschman
relationship: CREATED
work: The Strategy of Economic Development
</item 10>
=== END ITEM id=04dd00d039bc0132 ===

=== ITEM id=925ba4a4be9ffe8d ===
Gold sentences and facts:
Sentence 0: The Walking Dead Cover of The Walking Dead No. 1 . Art by Tony Moore . Publication information Publisher Image Comics Schedule Monthly Format Ongoing series Genre Post-apocalyptic Publication date October 2003 -- present No. of issues 178 Creative team Created by Robert Kirkman Tony Moore Written by Robert Kirkman Artist ( s ) Tony Moore ( # 1 -- 6 ) Charlie Adlard ( # 7 -- present ) Penciller ( s ) Tony Moore ( # 1 -- 6 ) Charlie Adlard ( # 7 -- present ) Inker ( s ) Tony Moore ( # 1 -- 6 ) Charlie Adlard ( # 7 -- present ) Stefano Gaudiano ( # 115 -- present ) Letterer ( s ) Robert Kirkman ( # 1 -- 19 ) Rus Wooton ( # 20 -- present ) Colorist ( s ) Tony Moore ( # 1 -- 5 ) Cliff Rathburn ( # 6 -- present ) Editor ( s ) Aubrey Sitterson ( # 55 -- 70 ) Sina Grace ( # 67 -- 96 ) Sean Mackiewicz ( # 97 -- present )
  Fact 0.0: The Walking Dead No. 1 has a cover with art by Tony Moore.
  Fact 0.1: The Walking Dead is published by Image Comics.
  Fact 0.2: The Walking Dead has a monthly publication schedule.
  Fact 0.3: The Walking Dead is an ongoing series.
  Fact 0.4: The Walking Dead is a post-apocalyptic comic.
  Fact 0.5: The Walking Dead was published from October 2003 to the present.
  Fact 0.6: The Walking Dead has 178 issues.
  Fact 0.7: The Walking Dead was created by Robert Kirkman and Tony Moore.
  Fact 0.8: Robert Kirkman wrote The Walking Dead.
  Fact 0.9: Tony Moore was the artist for issues 1 to 6 of The Walking Dead.
  Fact 0.10: Charlie Adlard was the artist from issue 7 of The Walking Dead to the present.
  Fact 0.11: Tony Moore was the penciller for issues 1 to 6 of The Walking Dead.
  Fact 0.12: Charlie Adlard was the penciller from issue 7 of The Walking Dead to the present.
  Fact 0.13: Tony Moore was the inker for issues 1 to 6 of The Walking Dead.
  Fact 0.14: Charlie Adlard was the inker from issue 7 of The Walking Dead to the present.
  Fact 0.15: Stefano Gaudiano was an inker of The Walking Dead from issue 115 to the present.
  Fact 0.16: Robert Kirkman was the letterer of The Walking Dead for issues 1 to 19.
  Fact 0.17: Rus Wooton was the letterer of The Walking Dead from issue 20 to the present.
  Fact 0.18: Tony Moore was the colorist of The Walking Dead for issues 1 to 5.
  Fact 0.19: Cliff Rathburn was the colorist of The Walking Dead from issue 6 to the present.
  Fact 0.20: Aubrey Sitterson was an editor of The Walking Dead for issues 55 to 70.
  Fact 0.21: Sina Grace was an editor of The Walking Dead for issues 67 to 96.
  Fact 0.22: Sean Mackiewicz was an editor of The Walking Dead from issue 97 to the present.
Sentence 1: The Walking Dead debuted in 2003 , published by Image Comics , with art by Tony Moore for the first six issues and Cliff Rathburn shading the art after issue five . Charlie Adlard took over as artist on issue # 7 , after he was approached by Kirkman . Moore also did the cover art for the first twenty - four issues and the first four trade paperbacks . The remainder were done by Adlard .
  Fact 1.0: The Walking Dead debuted in 2003.
  Fact 1.1: The Walking Dead was published by Image Comics.
  Fact 1.2: Tony Moore did the art for the first six issues of The Walking Dead.
  Fact 1.3: Cliff Rathburn shaded the art of The Walking Dead after issue five.
  Fact 1.4: Charlie Adlard took over as artist of The Walking Dead on issue 7.
  Fact 1.5: Robert Kirkman approached Charlie Adlard before Adlard took over as artist on issue 7.
  Fact 1.6: Tony Moore did the cover art for the first twenty-four issues of The Walking Dead.
  Fact 1.7: Tony Moore did the cover art for the first four trade paperbacks of The Walking Dead.
  Fact 1.8: Charlie Adlard did the remainder of the cover art of The Walking Dead.

Retrieved items:
<item 1>
The Walking Dead (comic book)
17 : Something To Fear 1 - 60706 - 615 - 7 November 21 , 2012 The Walking Dead # 97 -- 102 May 2012 -- September 2012 9 5 The Walking Dead Vol. 18 : What Comes
After 1 - 60706 - 687 - 4 June 18 , 2013 The Walking Dead # 103 -- 108 October 2012 -- March 2013 The Walking Dead Vol. 19 : March To War 1 - 60706 - 818 - 4
November 13 , 2013 The Walking Dead # 109 -- 114 April 2013 -- September 2013 10 The Walking Dead Vol. 20 : All Out War ( Part One ) 1 - 60706 - 882 - 6 March
11 , 2014 The Walking Dead # 115 -- 120 October 2013 -- January 2014 The Walking Dead Vol.
</item 1>

<item 2>
The Walking Dead (comic book)
2017 Jay Bonansinga Collected editions ( edit ) The series has so far been assembled into the following collections : Trade paperbacks ( edit ) The trade
paperbacks collect story arcs of six issues each , but contain only the story and none of the original cover art from the comics . Each paperback follows the
convention of having a three - word title . The zombies in the cover art for each paperback form part of a larger image if placed end to end . Each story arc is
re-released in hardcover books containing two , omnibus editions containing four , and compendiums containing eight story arcs . Title ISBN Release Date
Collected Material Issues Published Hardcover Book Number Omnibus Number Compendium Number The Walking Dead Vol. 1 : Days Gone Bye 1 - 58240 - 358 - 9 May 12 ,
2004 The Walking Dead # 1 -- 6 October 2003 -- March 2004 The Walking Dead Vol.
</item 2>

<item 3>
The Walking Dead (comic book)
2005 Collects # 1 -- 24 The Walking Dead : Volume 2 Deluxe HC 1 - 60706 - 029 - 9 February 17 , 2009 Collects # 25 -- 48 The Walking Dead : Volume 3 Deluxe HC 1
- 60706 - 330 - 1 February 2 , 2011 Collects # 49 -- 72 The Walking Dead : Volume 4 Deluxe HC 1 - 60706 - 616 - 5 December 12 , 2012 Collects # 73 -- 96 The
Walking Dead : Volume 5 Deluxe HC 1 - 63215 - 042 - 5 November 19 , 2014 Collects # 97 -- 120 The Walking Dead : Volume 6 Deluxe HC 1 - 63215 - 521 - 4 December
2 , 2015 Collects # 121 -- 144 The Walking Dead : Volume 7 Deluxe HC 1 - 53430 - 335 - 9 November 7 , 2017 Collects # 145 -- 168 Compendium editions ( edit )
Softcover compendium editions collect 48 comic issues each .
</item 3>

<item 4>
The Walking Dead (comic book)
21 : All Out War ( Part Two ) 1 - 63215 - 030 - 1 July 29 , 2014 The Walking Dead # 121 -- 126 February 2014 -- April 2014 11 6 The Walking Dead Vol. 22 : A New
Beginning 1 - 63215 - 041 - 7 November 5 , 2014 The Walking Dead # 127 -- 132 May 2014 -- October 2014 The Walking Dead Vol. 23 : Whispers Into Screams 1 -
63215 - 258 - 4 May 12 , 2015 The Walking Dead # 133 -- 138 October 2014 -- March 2015 12 The Walking Dead Vol. 24 : Life And Death 1 - 63215 - 402 - 1 August
26 , 2015 The Walking Dead # 139 -- 144 March 2015 -- July 2015 The Walking Dead Vol.
</item 4>

<item 5>
The Walking Dead (comic book)
29 : Lines We Cross 1 - 53430 - 497 - 5 March 13 , 2018 The Walking Dead # 169 -- 174 July 2017 -- December 2017 15 8 Hardcovers ( edit ) All hardcovers contain
the contents of the comics , including the covers , and in some cases bonus material . The books ' trim size is larger than the paperbacks . Each hardcover
contains two story arcs from the series . Signed versions of the books are available , each limited to 310 pieces . Title ISBN Release Date Collected Material
Cover Character The Walking Dead : Book One 978 - 1582406190 July 19 , 2006 The Walking Dead # 1 -- 12 Rick The Walking Dead : Book Two 978 - 1582406985 March 7
, 2007 The Walking Dead # 13 -- 24 Michonne The Walking Dead : Book Three 978 - 1582408255 December 19 , 2007 The Walking Dead # 25 -- 36 The Governor The
Walking Dead : Book Four 978 - 1607060000 October 29 ,
</item 5>

<item 6>
The Walking Dead (comic book)
12 : Life Among Them 1 - 60706 - 254 - 2 August 3 , 2010 The Walking Dead # 67 -- 72 November 2009 -- May 2010 The Walking Dead Vol. 13 : Too Far Gone 1 - 60706
- 329 - 8 November 23 , 2010 The Walking Dead # 73 -- 78 June 2010 -- October 2010 7 The Walking Dead Vol. 14 : No Way Out 1 - 60706 - 392 - 1 June 22 , 2011
The Walking Dead # 79 -- 84 November 2010 -- April 2011 The Walking Dead Vol. 15 : We Find Ourselves 1 - 60706 - 440 - 5 December 27 , 2011 The Walking Dead #
85 -- 90 May 2011 -- October 2011 8 The Walking Dead Vol. 16 : A Larger World 1 - 60706 - 559 - 2 June 6 , 2012 The Walking Dead # 91 -- 96 November 2011 --
April 2012 The Walking Dead Vol.
</item 6>

<item 7>
The Walking Dead (comic book)
It focuses on Rick Grimes , a Kentucky deputy who is shot in the line of duty and awakens from a coma in a zombie apocalypse that has resulted in a state - wide
quarantine . After joining with other survivors , including his loved ones , he gradually takes on the role of leader of a community as it struggles to survive
the zombie apocalypse . First issued in 2003 by publisher Image Comics , the comic is written by Kirkman with art by Moore ( issues No. 1 -- 6 ) and Charlie
Adlard ( issue No. 7 onward ) . Moore continued to do the covers through issue No. 24 . The Walking Dead received the 2007 and 2010 Eisner Award for Best
Continuing Series at San Diego Comic - Con International . The series was adapted into the AMC television series The Walking Dead , which premiered in 2010 .
The television program loosely follows the storyline of the comic book .
</item 7>

<item 8>
The Walking Dead (comic book)
Charlie Adlard took over as artist on issue # 7 , after he was approached by Kirkman . Moore also did the cover art for the first twenty - four issues and the
first four trade paperbacks . The remainder were done by Adlard . When the television series premiered in October 2010 , Image Comics announced The Walking Dead
Weekly . The first 52 issues of the series began to be reprinted on January 5 , 2011 , with one issue per week for a year . The series is periodically
re-published in trade paperbacks which contain six issues each , hardcover books with twelve issues and occasional bonus material , omnibus editions of twenty -
four issues , and compendium editions of forty - eight issues . Story arcs ( edit ) Volume 1 : Days Gone Bye ( Issues 1 -- 6 ) Rick Grimes , a sheriff 's deputy
from Kentucky , is wounded in the line of duty and emerges from a coma to find the world overrun by the undead . He returns home to find his house ransacked and
his wife and son gone .
</item 8>

<item 9>
The Walking Dead (comic book)
2008 The Walking Dead # 37 -- 48 Lori and Judith The Walking Dead : Book Five 978 - 1607061717 May 5 , 2010 The Walking Dead # 49 -- 60 Abraham The Walking Dead
: Book Six 978 - 1607063278 October 26 , 2010 The Walking Dead # 61 -- 72 Carl The Walking Dead : Book Seven 978 - 1607064398 October 18 , 2011 The Walking Dead
# 73 -- 84 Rick The Walking Dead : Book Eight 978 - 1607065937 September 26 , 2012 The Walking Dead # 85 -- 96 Andrea The Walking Dead : Book Nine 978 -
1607067986 September 17 , 2013 The Walking Dead # 97 -- 108 Negan The Walking Dead : Book Ten 978 - 1632150349 September 3 , 2014 The Walking Dead # 109 -- 120
Ezekiel The Walking Dead : Book Eleven 978 - 1632152718 February 25 ,
</item 9>

<item 10>
The Walking Dead (comic book)
The Walking Dead ( comic book ) - Wikipedia The Walking Dead ( comic book ) Jump to : navigation , search The Walking Dead Cover of The Walking Dead No. 1 . Art
by Tony Moore . Publication information Publisher Image Comics Schedule Monthly Format Ongoing series Genre Post-apocalyptic Publication date October 2003 --
present No.
</item 10>
=== END ITEM id=925ba4a4be9ffe8d ===

=== ITEM id=7f7e5f15209afc6f ===
Gold sentences and facts:
Sentence 0: The Second Great Awakening occurred in several episodes and over different denominations ; however , the revivals were very similar . As the most effective form of evangelizing during this period , revival meetings cut across geographical boundaries , and the movement quickly spread throughout Kentucky , Tennessee and southern Ohio . Each denomination had assets that allowed it to thrive on the frontier . The Methodists had an efficient organization that depended on itinerant ministers , known as circuit riders , who sought out people in remote frontier locations . The circuit riders came from among the common people , which helped them establish rapport with the frontier families they hoped to convert .
  Fact 0.0: The Second Great Awakening occurred in several episodes.
  Fact 0.1: The Second Great Awakening occurred over different denominations.
  Fact 0.2: The revivals of the Second Great Awakening were very similar to each other.
  Fact 0.3: Revival meetings were the most effective form of evangelizing during this period.
  Fact 0.4: Revival meetings cut across geographical boundaries.
  Fact 0.5: The revival movement quickly spread throughout Kentucky, Tennessee and southern Ohio.
  Fact 0.6: Each denomination had assets that allowed it to thrive on the frontier.
  Fact 0.7: The Methodists had an efficient organization.
  Fact 0.8: The Methodist organization depended on itinerant ministers known as circuit riders.
  Fact 0.9: Circuit riders sought out people in remote frontier locations.
  Fact 0.10: Circuit riders came from among the common people.
  Fact 0.11: Coming from among the common people helped circuit riders establish rapport with the frontier families they hoped to convert.
Sentence 1: The revivals enrolled millions of new members in existing evangelical denominations and led to the formation of new denominations . Many converts believed that the Awakening heralded a new millennial age . The Second Great Awakening stimulated the establishment of many reform movements designed to remedy the evils of society before the anticipated Second Coming of Jesus Christ .
  Fact 1.0: The revivals enrolled millions of new members in existing evangelical denominations.
  Fact 1.1: The revivals led to the formation of new denominations.
  Fact 1.2: Many converts believed that the Awakening heralded a new millennial age.
  Fact 1.3: The Second Great Awakening stimulated the establishment of many reform movements.
  Fact 1.4: The reform movements were designed to remedy the evils of society before the anticipated Second Coming of Jesus Christ.
Sentence 2: In the newly settled frontier regions , the revival was implemented through camp meetings . These often provided the first encounter for some settlers with organized religion , and they were important as social venues . The camp meeting was a religious service of several days ' length with preachers . Settlers in thinly populated areas gathered at the camp meeting for fellowship as well as worship . The sheer exhilaration of participating in a religious revival with crowds of hundreds and perhaps thousands of people inspired the dancing , shouting , and singing associated with these events . The revivals followed an arc of great emotional power , with an emphasis of the individual 's sins and need to turn to Christ , restored by a sense of personal salvation . Upon their return home , most converts joined or created small local churches , which grew rapidly . The Second Great Awakening marked a religious transition in society in America . Many Americans from the Calvinist sect emphasized man 's inability to save themselves and that their only way to be saved was from grace from God .
  Fact 2.0: In the newly settled frontier regions, the revival was implemented through camp meetings.
  Fact 2.1: Camp meetings often provided the first encounter for some settlers with organized religion.
  Fact 2.2: Camp meetings were important as social venues.
  Fact 2.3: A camp meeting was a religious service of several days' length with preachers.
  Fact 2.4: Settlers in thinly populated areas gathered at the camp meeting for fellowship as well as worship.
  Fact 2.5: Participating in a religious revival with crowds of hundreds and perhaps thousands of people was sheer exhilaration.
  Fact 2.6: That exhilaration inspired the dancing, shouting, and singing associated with camp meeting events.
  Fact 2.7: The revivals followed an arc of great emotional power.
  Fact 2.8: The revivals emphasized the individual's sins and need to turn to Christ.
  Fact 2.9: The revivals were restored by a sense of personal salvation.
  Fact 2.10: Upon their return home, most converts joined or created small local churches.
  Fact 2.11: The small local churches grew rapidly.
  Fact 2.12: The Second Great Awakening marked a religious transition in society in America.
  Fact 2.13: Many Americans from the Calvinist sect emphasized man's inability to save themselves.
  Fact 2.14: Many Americans from the Calvinist sect emphasized that their only way to be saved was by grace from God.

Retrieved items:
<item 1>
Second Great Awakening
the Second reflected Romanticism characterized by enthusiasm , emotion , and an appeal to the super-natural . It rejected the skepticism , deism , and
rationalism left over from the Enlightenment . At about the same time , similar movements flourished in Europe . Pietism was sweeping German countries .
Evangelicalism was waxing strong in England . The Second Great Awakening occurred in several episodes and over different denominations ; however , the revivals
were very similar . As the most effective form of evangelizing during this period , revival meetings cut across geographical boundaries , and the movement
quickly spread throughout Kentucky , Tennessee and southern Ohio . Each denomination had assets that allowed it to thrive on the frontier . The Methodists had
an efficient organization that depended on itinerant ministers , known as circuit riders , who sought out people in remote frontier locations . The circuit
riders came from among the common people , which helped them establish rapport with the frontier families they hoped to convert . Theology ( edit ) Main article
: Postmillennialism Postmillennialism theology dominated American Protestantism in the first half of the 19th century .
</item 1>

<item 2>
Second Great Awakening
West ( edit ) Main article : Revival of 1800 In the newly settled frontier regions , the revival was implemented through camp meetings . These often provided
the first encounter for some settlers with organized religion , and they were important as social venues . The camp meeting was a religious service of several
days ' length with preachers . Settlers in thinly populated areas gathered at the camp meeting for fellowship as well as worship . The sheer exhilaration of
participating in a religious revival with crowds of hundreds and perhaps thousands of people inspired the dancing , shouting , and singing associated with these
events . The revivals followed an arc of great emotional power , with an emphasis of the individual 's sins and need to turn to Christ , restored by a sense of
personal salvation . Upon their return home , most converts joined or created small local churches , which grew rapidly . The Second Great Awakening marked a
religious transition in society in America . Many Americans from the Calvinist sect emphasized man 's inability to save themselves and that their only way to be
saved was from grace from God . The Revival of 1800 in Logan County , Kentucky , began as a traditional Presbyterian sacramental occasion .
</item 2>

<item 3>
Second Great Awakening
Contents ( hide ) 1 Spread of revivals 1.1 Background 1.2 Theology 1.3 Burned - over district 1.4 West and Tidewater South 1.5 West 1.6 Church membership soars
2 Subgroups 2.1 Adventism 2.2 Holiness movement 2.3 Restoration Movement 3 Culture and society 4 Slaves and free Africans 5 Women 6 Prominent figures 7
Political implications 8 See also 9 References 10 Further reading 10.1 Historiography Spread of revivals ( edit ) Part of a series on Baptists Background ( show
) Christianity Protestantism Puritanism Anabaptism Doctrine ( show ) Priesthood of all believers Individual soul liberty Separation of church and state Sola
scriptura Congregationalism Ordinances Offices Confessions Key figures ( show ) John Smyth Thomas Helwys Roger Williams John Clarke John Bunyan Shubal Stearns
Andrew Fuller Charles Spurgeon D.N. Jackson James Robinson Graves William Bullein Johnson William Carey Luther Rice Martin Luther King Jr .
</item 3>

<item 4>
Second Great Awakening
The influence of the Awakening continued in the form of more secular movements . In the midst of shifts in theology and church polity , American Christians
began progressive movements to reform society during this period . Known commonly as antebellum reform , this phenomenon included reforms in against the
consumption of alcohol , for women 's rights and abolition of slavery , and a multitude of other issues faced by society . The religious enthusiasm of the
Second Great Awakening was echoed by the new political enthusiasm of the Second Party System . More active participation in politics by more segments of the
population brought religious and moral issues into the political sphere . The spirit of evangelical humanitarian reforms was carried on in the antebellum Whig
party . Historians stress the understanding common among participants of reform as being a part of God 's plan . As a result , local churches saw their roles in
society in purifying the world through the individuals to whom they could bring salvation , and through changes in the law and the creation of institutions .
Interest in transforming the world was applied to mainstream political action , as temperance activists , antislavery advocates , and proponents of other
variations of reform sought to implement their beliefs into national politics .
</item 4>

<item 5>
Second Great Awakening
The Restoration Movement began during , and was greatly influenced by , the Second Great Awakening . While the leaders of one of the two primary groups making
up this movement , Thomas Campbell and Alexander Campbell , resisted what they saw as the spiritual manipulation of the camp meetings , the revivals contributed
to the development of the other major branch , led by Barton W. Stone . The Southern phase of the Awakening `` was an important matrix of Barton Stone 's reform
movement '' and shaped the evangelistic techniques used by both Stone and the Campbells . Culture and society ( edit ) Efforts to apply Christian teaching to
the resolution of social problems presaged the Social Gospel of the late 19th century . Converts were taught that to achieve salvation they needed not just to
repent personal sin but also work for the moral perfection of society , which meant eradicating sin in all its forms . Thus , evangelical converts were leading
figures in a variety of 19th century reform movements . Congregationalists set up missionary societies to evangelize the western territory of the northern tier
. Members of these groups acted as apostles for the faith , and also as educators and exponents of northeastern urban culture .
</item 5>

<item 6>
Second Great Awakening
Second Great Awakening - wikipedia Second Great Awakening Jump to : navigation , search An engraving of a Methodist camp meeting in 1819 ( Library of Congress )
. Great Awakening First ( c. 1730 -- 1755 ) Second ( c. 1790 -- 1840 ) Third ( c. 1855 -- 1900 ) Fourth ( c. 1960 -- 1980 ) The Second Great Awakening was a
Protestant religious revival during the early 19th century in the United States . The movement began around 1790 , gained momentum by 1800 and , after 1820 ,
membership rose rapidly among Baptist and Methodist congregations whose preachers led the movement . It was past its peak by the late 1850s . The Second Great
Awakening reflected Romanticism characterized by enthusiasm , emotion , and an appeal to the super-natural . It rejected the skeptical rationalism and deism of
the Enlightenment . The revivals enrolled millions of new members in existing evangelical denominations and led to the formation of new denominations . Many
converts believed that the Awakening heralded a new millennial age .
</item 6>

<item 7>
Second Great Awakening
a re-emphasis on Wesleyan teachings on sanctification emerged during the Second Great Awakening , leading to a distinction between Mainline Methodism and
Holiness churches . Restoration movement ( edit ) Main article : Restoration Movement See also : Restorationism The idea of restoring a `` primitive '' form of
Christianity grew in popularity in the U.S. after the American Revolution . This desire to restore a purer form of Christianity without an elaborate hierarchy
contributed to the development of many groups during the Second Great Awakening , including the Mormons , Baptists and Shakers . Several factors made the
restoration sentiment particularly appealing during this time period : To immigrants in the early 19th century , the land in the United States seemed pristine ,
edenic and undefiled -- `` the perfect place to recover pure , uncorrupted and original Christianity '' -- and the tradition - bound European churches seemed
out of place in this new setting . A primitive faith based on the Bible alone promised a way to sidestep the competing claims of the many denominations
available and for congregations to find assurance of being right without the security of an established national church . The Restoration Movement began during
, and was greatly influenced by , the Second Great Awakening .
</item 7>

<item 8>
Second Great Awakening
Many converts believed that the Awakening heralded a new millennial age . The Second Great Awakening stimulated the establishment of many reform movements
designed to remedy the evils of society before the anticipated Second Coming of Jesus Christ . Historians named the Second Great Awakening in the context of the
First Great Awakening of the 1730s and 1740s and of the Third Great Awakening of the late 1850s to early 1900s . These revivals were part of a much larger
Romantic religious movement that was sweeping across Europe at the time , mainly throughout England , Scotland , and Germany .
</item 8>

<item 9>
Second Great Awakening
The Second Great Awakening served as an `` organizing process '' that created `` a religious and educational infrastructure '' across the western frontier that
encompassed social networks , a religious journalism that provided mass communication , and church - related colleges . Publication and education societies
promoted Christian education ; most notable among them was the American Bible Society , founded in 1816 . Women made up a large part of these voluntary
societies . The Female Missionary Society and the Maternal Association , both active in Utica , NY , were highly organized and financially sophisticated women
's organizations responsible for many of the evangelical converts of the New York frontier . There were also societies that broadened their focus from
traditional religious concerns to larger societal ones . These organizations were primarily sponsored by affluent women . They did not stem entirely from the
Second Great Awakening , but the revivalist doctrine and the expectation that one 's conversion would lead to personal action accelerated the role of women 's
social benevolence work . Social activism influenced abolition groups and supporters of the Temperance movement . They began efforts to reform prisons and care
for the handicapped and mentally ill . They believed in the perfectibility of people and were highly moralistic in their endeavors .
</item 9>

<item 10>
Second Great Awakening
While Protestant religion had previously played an important role on the American political scene , the Second Great Awakening strengthened the role it would
play . See also ( edit ) Baptist portal Methodism portal Calvinism portal Seventh - day Adventist Church portal Latter Day Saints portal LDS Church portal
Advent Christian Church Christian revival Christianity in the 19th century The Church of Jesus Christ of Latter - day Saints Cumberland Presbyterian Church
Ethnocultural politics in the United States Holiness movement Restoration Movement Seventh - day Adventist Church References ( edit ) ^ Jump up to : Timothy L.
Smith , Revivalism and Social Reform : American Protestantism on the Eve of the Civil War ( 1957 ) . Jump up ^ Heyrman , Christine Leigh . `` The First Great
Awakening '' . Divining America , TeacherServe . National Humanities Center . Jump up ^ Henry B. Clark ( 1982 ) . Freedom of Religion in America : Historical
Roots , Philosophical Concepts , Contemporary Problems . Transaction Publishers . p. 16 . ^ Jump up to : Nancy Cott , `` Young Women in the Second Great
Awakening in New England , '' Feminist Studies , ( 1975 ) , 3 # 1 p.
</item 10>
=== END ITEM id=7f7e5f15209afc6f ===

=== ITEM id=c77e248d8033d96e ===
Gold sentences and facts:
Sentence 0: No . overall No. in season Title Directed by Written by Original air date Prod . code U.S. viewers ( millions ) 70 `` The Flash Reborn '' Glen Winter Story by : Andrew Kreisberg Teleplay by : Todd Helbing & Eric Wallace October 10 , 2017 ( 2017 - 10 - 10 ) T27. 13401 2.84 71 `` Mixed Signals '' Alexandra La Roche Jonathan Butler & Gabriel Garza October 17 , 2017 ( 2017 - 10 - 17 ) T27. 13402 2.54 72 `` Luck Be a Lady '' Armen V. Kevorkian Sam Chalsen & Judalina Neira October 24 , 2017 ( 2017 - 10 - 24 ) T27. 13403 2.62 73 `` Elongated Journey Into Night '' Tom Cavanagh Sterling Gates & Thomas Pound October 31 , 2017 ( 2017 - 10 - 31 ) T27. 13404 1.99 74 5 `` Girls Night Out '' Laura Belsey Lauren Certo & Kristen Kim November 7 , 2017 ( 2017 - 11 - 07 ) T27. 13405 2.38 75 6 `` When Harry Met Harry ... '' Brent Crowell Jonathan Butler & Gabriel Garza November 14 , 2017 ( 2017 - 11 - 14 ) T27. 13406 2.46 76 7 `` Therefore I Am '' David McWhirter Eric Wallace & Thomas Pound November 21 , 2017 ( 2017 - 11 - 21 ) T27. 13407 2.20 77 8 `` Crisis on Earth - X , Part 3 '' Dermott Downs Story by : Andrew Kreisberg & Marc Guggenheim Teleplay by : Todd Helbing November 28 , 2017 ( 2017 - 11 - 28 ) T27. 13408 2.82 78 9 `` Do n't Run '' Stefan Pleszczynski Sam Chalsen & Judalina Neira December 5 , 2017 ( 2017 - 12 - 05 ) T27. 13409 2.22 79 10 `` The Trial of The Flash '' Philip Chipera Lauren Certo & Kristen Kim January 16 , 2018 ( 2018 - 01 - 16 ) T27. 13410 2.51 80 11 `` The Elongated Knight Rises '' Alexandra La Roche Sterling Gates & Thomas Pound January 23 , 2018 ( 2018 - 01 - 23 ) T27. 13411 2.12 81 12 `` Honey , I Shrunk Team Flash '' Chris Peppe Sam Chalsen & Judalina Neira January 30 , 2018 ( 2018 - 01 - 30 ) T27. 13412 2.60 82 13 `` True Colors '' Tara Nicole Weyr Jonathan Butler & Gabriel Garza February 6 , 2018 ( 2018 - 02 - 06 ) T27. 13413 2.28 83 14 `` Subject 9 '' Ralph Hemecker Mike Alber & Gabe Snyder February 27 , 2018 ( 2018 - 02 - 27 ) T27. 13414 2.12 84 15 `` Enter Flashtime '' Gregory Smith Todd Helbing & Sterling Gates March 6 , 2018 ( 2018 - 03 - 06 ) T27. 13415 2.04 85 16 `` Run , Iris , Run '' Harry Jierjian Eric Wallace March 13 , 2018 ( 2018 - 03 - 13 ) T27. 13416 2.09 86 17 `` Null and Annoyed '' Kevin Smith Lauren Certo & Kristen Kim April 10 , 2018 ( 2018 - 04 - 10 ) T27. 13417 1.82 87 18 `` Lose Yourself '' Hanelle Culpepper Jonathan Butler & Gabriel Garza April 17 , 2018 ( 2018 - 04 - 17 ) T27. 13418 1.88 88 19 `` Fury Rogue '' Rachel Talalay Joshua V. Gilbert & Jeff Hersh April 24 , 2018 ( 2018 - 04 - 24 ) TBA 1.90 89 20 `` Therefore She Is '' Rob J. Greenlea Sterling Gates & Thomas Pound May 1 , 2018 ( 2018 - 05 - 01 ) TBA TBD 90 21 `` Harry and the Harrisons '' Kevin Mock Judalina Neira & Lauren Certo May 8 , 2018 ( 2018 - 05 - 08 ) TBA TBD 91 22 `` Think Fast '' Viet Nguyen Sam Chalsen & Kristen Kim May 15 , 2018 ( 2018 - 05 - 15 ) TBA TBD 92 23 `` We Are the Flash '' TBA TBA May 22 , 2018 ( 2018 - 05 - 22 ) TBA TBD
  Fact 0.0: Episode 70 overall is titled The Flash Reborn.
  Fact 0.1: The Flash Reborn was directed by Glen Winter.
  Fact 0.2: The Flash Reborn had story by Andrew Kreisberg and teleplay by Todd Helbing and Eric Wallace.
  Fact 0.3: The Flash Reborn aired October 10, 2017, with production code T27.13401 and 2.84 million U.S. viewers.
  Fact 0.4: Episode 71 overall is titled Mixed Signals.
  Fact 0.5: Mixed Signals was directed by Alexandra La Roche.
  Fact 0.6: Mixed Signals was written by Jonathan Butler and Gabriel Garza.
  Fact 0.7: Mixed Signals aired October 17, 2017, with production code T27.13402 and 2.54 million U.S. viewers.
  Fact 0.8: Episode 72 overall is titled Luck Be a Lady.
  Fact 0.9: Luck Be a Lady was directed by Armen V. Kevorkian.
  Fact 0.10: Luck Be a Lady was written by Sam Chalsen and Judalina Neira.
  Fact 0.11: Luck Be a Lady aired October 24, 2017, with production code T27.13403 and 2.62 million U.S. viewers.
  Fact 0.12: Episode 73 overall is titled Elongated Journey Into Night.
  Fact 0.13: Elongated Journey Into Night was directed by Tom Cavanagh.
  Fact 0.14: Elongated Journey Into Night was written by Sterling Gates and Thomas Pound.
  Fact 0.15: Elongated Journey Into Night aired October 31, 2017, with production code T27.13404 and 1.99 million U.S. viewers.
  Fact 0.16: Episode 74 overall is episode 5 of its season and is titled Girls Night Out.
  Fact 0.17: Girls Night Out was directed by Laura Belsey.
  Fact 0.18: Girls Night Out was written by Lauren Certo and Kristen Kim.
  Fact 0.19: Girls Night Out aired November 7, 2017, with production code T27.13405 and 2.38 million U.S. viewers.
  Fact 0.20: Episode 75 overall is episode 6 of its season and is titled When Harry Met Harry ...
  Fact 0.21: When Harry Met Harry ... was directed by Brent Crowell.
  Fact 0.22: When Harry Met Harry ... was written by Jonathan Butler and Gabriel Garza.
  Fact 0.23: When Harry Met Harry ... aired November 14, 2017, with production code T27.13406 and 2.46 million U.S. viewers.
  Fact 0.24: Episode 76 overall is episode 7 of its season and is titled Therefore I Am.
  Fact 0.25: Therefore I Am was directed by David McWhirter.
  Fact 0.26: Therefore I Am was written by Eric Wallace and Thomas Pound.
  Fact 0.27: Therefore I Am aired November 21, 2017, with production code T27.13407 and 2.20 million U.S. viewers.
  Fact 0.28: Episode 77 overall is episode 8 of its season and is titled Crisis on Earth - X, Part 3.
  Fact 0.29: Crisis on Earth - X, Part 3 was directed by Dermott Downs.
  Fact 0.30: Crisis on Earth - X, Part 3 had story by Andrew Kreisberg and Marc Guggenheim and teleplay by Todd Helbing.
  Fact 0.31: Crisis on Earth - X, Part 3 aired November 28, 2017, with production code T27.13408 and 2.82 million U.S. viewers.
  Fact 0.32: Episode 78 overall is episode 9 of its season and is titled Don't Run.
  Fact 0.33: Don't Run was directed by Stefan Pleszczynski.
  Fact 0.34: Don't Run was written by Sam Chalsen and Judalina Neira.
  Fact 0.35: Don't Run aired December 5, 2017, with production code T27.13409 and 2.22 million U.S. viewers.
  Fact 0.36: Episode 79 overall is episode 10 of its season and is titled The Trial of The Flash.
  Fact 0.37: The Trial of The Flash was directed by Philip Chipera.
  Fact 0.38: The Trial of The Flash was written by Lauren Certo and Kristen Kim.
  Fact 0.39: The Trial of The Flash aired January 16, 2018, with production code T27.13410 and 2.51 million U.S. viewers.
  Fact 0.40: Episode 80 overall is episode 11 of its season and is titled The Elongated Knight Rises.
  Fact 0.41: The Elongated Knight Rises was directed by Alexandra La Roche.
  Fact 0.42: The Elongated Knight Rises was written by Sterling Gates and Thomas Pound.
  Fact 0.43: The Elongated Knight Rises aired January 23, 2018, with production code T27.13411 and 2.12 million U.S. viewers.
  Fact 0.44: Episode 81 overall is episode 12 of its season and is titled Honey, I Shrunk Team Flash.
  Fact 0.45: Honey, I Shrunk Team Flash was directed by Chris Peppe.
  Fact 0.46: Honey, I Shrunk Team Flash was written by Sam Chalsen and Judalina Neira.
  Fact 0.47: Honey, I Shrunk Team Flash aired January 30, 2018, with production code T27.13412 and 2.60 million U.S. viewers.
  Fact 0.48: Episode 82 overall is episode 13 of its season and is titled True Colors.
  Fact 0.49: True Colors was directed by Tara Nicole Weyr.
  Fact 0.50: True Colors was written by Jonathan Butler and Gabriel Garza.
  Fact 0.51: True Colors aired February 6, 2018, with production code T27.13413 and 2.28 million U.S. viewers.
  Fact 0.52: Episode 83 overall is episode 14 of its season and is titled Subject 9.
  Fact 0.53: Subject 9 was directed by Ralph Hemecker.
  Fact 0.54: Subject 9 was written by Mike Alber and Gabe Snyder.
  Fact 0.55: Subject 9 aired February 27, 2018, with production code T27.13414 and 2.12 million U.S. viewers.
  Fact 0.56: Episode 84 overall is episode 15 of its season and is titled Enter Flashtime.
  Fact 0.57: Enter Flashtime was directed by Gregory Smith.
  Fact 0.58: Enter Flashtime was written by Todd Helbing and Sterling Gates.
  Fact 0.59: Enter Flashtime aired March 6, 2018, with production code T27.13415 and 2.04 million U.S. viewers.
  Fact 0.60: Episode 85 overall is episode 16 of its season and is titled Run, Iris, Run.
  Fact 0.61: Run, Iris, Run was directed by Harry Jierjian.
  Fact 0.62: Run, Iris, Run was written by Eric Wallace.
  Fact 0.63: Run, Iris, Run aired March 13, 2018, with production code T27.13416 and 2.09 million U.S. viewers.
  Fact 0.64: Episode 86 overall is episode 17 of its season and is titled Null and Annoyed.
  Fact 0.65: Null and Annoyed was directed by Kevin Smith.
  Fact 0.66: Null and Annoyed was written by Lauren Certo and Kristen Kim.
  Fact 0.67: Null and Annoyed aired April 10, 2018, with production code T27.13417 and 1.82 million U.S. viewers.
  Fact 0.68: Episode 87 overall is episode 18 of its season and is titled Lose Yourself.
  Fact 0.69: Lose Yourself was directed by Hanelle Culpepper.
  Fact 0.70: Lose Yourself was written by Jonathan Butler and Gabriel Garza.
  Fact 0.71: Lose Yourself aired April 17, 2018, with production code T27.13418 and 1.88 million U.S. viewers.
  Fact 0.72: Episode 88 overall is episode 19 of its season and is titled Fury Rogue.
  Fact 0.73: Fury Rogue was directed by Rachel Talalay.
  Fact 0.74: Fury Rogue was written by Joshua V. Gilbert and Jeff Hersh.
  Fact 0.75: Fury Rogue aired April 24, 2018, with production code TBA and 1.90 million U.S. viewers.
  Fact 0.76: Episode 89 overall is episode 20 of its season and is titled Therefore She Is.
  Fact 0.77: Therefore She Is was directed by Rob J. Greenlea.
  Fact 0.78: Therefore She Is was written by Sterling Gates and Thomas Pound.
  Fact 0.79: Therefore She Is aired May 1, 2018, with production code TBA and viewer count TBD.
  Fact 0.80: Episode 90 overall is episode 21 of its season and is titled Harry and the Harrisons.
  Fact 0.81: Harry and the Harrisons was directed by Kevin Mock.
  Fact 0.82: Harry and the Harrisons was written by Judalina Neira and Lauren Certo.
  Fact 0.83: Harry and the Harrisons aired May 8, 2018, with production code TBA and viewer count TBD.
  Fact 0.84: Episode 91 overall is episode 22 of its season and is titled Think Fast.
  Fact 0.85: Think Fast was directed by Viet Nguyen.
  Fact 0.86: Think Fast was written by Sam Chalsen and Kristen Kim.
  Fact 0.87: Think Fast aired May 15, 2018, with production code TBA and viewer count TBD.
  Fact 0.88: Episode 92 overall is episode 23 of its season and is titled We Are the Flash.
  Fact 0.89: The director and writers of We Are the Flash are listed as TBA.
  Fact 0.90: We Are the Flash aired May 22, 2018, with production code TBA and viewer count TBD.

Retrieved items:
<item 1>
The Flash is a CW action series developed by Greg Berlanti, Andrew Kreisberg and Geoff Johns, spun off from Arrow. Grant Gustin stars as Barry Allen, a Central
City crime-scene investigator who gains super-speed and fights metahuman criminals. The episode list covers four 23-episode seasons, beginning in 2014; seasons
three and four premiered in 2016 and 2017. By April 24, 2018, 88 episodes had aired, with
</item 1>

<item 2>
List of The Flash episodes
08 1.0 2.20 2.3 6.28 12 `` Crazy for You '' February 3 , 2015 1.3 / 4 3.60 0.9 N / A 2.2 N / A 13 `` The Nuclear Man '' February 10 , 2015 1.5 / 5 3.66 N / A N
/ A N / A N / A 14 `` Fallout '' February 17 , 2015 1.5 / 4 4.01 0.9 N / A 2.4 N / A 15 `` Out of Time '' March 17 , 2015 1.3 / 5 3.69 1.0 2.34 2.3 6.09 16 ``
Rogue Time '' March 24 , 2015 1.2 / 4 3.33 1.0 2.37 2.2 5.69 17 `` Tricksters '' March 31 , 2015 1.3 / 4 3.67 1.0 2.43 2.
</item 2>

<item 3>
List of The Flash episodes
13410 2.51 80 11 `` The Elongated Knight Rises '' Alexandra La Roche Sterling Gates & Thomas Pound January 23 , 2018 ( 2018 - 01 - 23 ) T27. 13411 2.12 81 12 ``
Honey , I Shrunk Team Flash '' Chris Peppe Sam Chalsen & Judalina Neira January 30 , 2018 ( 2018 - 01 - 30 ) T27. 13412 2.60 82 13 `` True Colors '' Tara Nicole
Weyr Jonathan Butler & Gabriel Garza February 6 , 2018 ( 2018 - 02 - 06 ) T27. 13413 2.28 83 14 `` Subject 9 '' Ralph Hemecker Mike Alber & Gabe Snyder February
27 , 2018 ( 2018 - 02 - 27 ) T27. 13414 2.12 84 15 `` Enter Flashtime '' Gregory Smith Todd Helbing & Sterling Gates March 6 , 2018 ( 2018 - 03 - 06 ) T27.
</item 3>

<item 4>
List of The Flash episodes
13415 2.04 85 16 `` Run , Iris , Run '' Harry Jierjian Eric Wallace March 13 , 2018 ( 2018 - 03 - 13 ) T27. 13416 2.09 86 17 `` Null and Annoyed '' Kevin Smith
Lauren Certo & Kristen Kim April 10 , 2018 ( 2018 - 04 - 10 ) T27. 13417 1.82 87 18 `` Lose Yourself '' Hanelle Culpepper Jonathan Butler & Gabriel Garza April
17 , 2018 ( 2018 - 04 - 17 ) T27.
</item 4>

<item 5>
List of The Flash episodes
2014 1.4 / 4 3.73 0.9 N / A 2.3 N / A 7 `` Power Outage '' November 25 , 2014 1.4 / 4 3.47 0.9 2.41 2.3 5.88 8 `` Flash vs. Arrow '' December 2 , 2014 1.6 / 5
4.34 1.0 2.31 2.6 6.65 9 `` The Man in the Yellow Suit '' December 9 , 2014 1.5 / 5 4.66 0.9 2.06 2.4 6.72 10 `` Revenge of the Rogues '' January 20 , 2015 1.4
/ 5 3.87 1.0 2.41 2.4 6.27 11 `` The Sound and the Fury '' January 27 , 2015 1.3 / 4 4.08 1.0 2.20 2.3 6.
</item 5>

<item 6>
List of The Flash episodes
8 8 `` Flash vs. Arrow '' Glen Winter Story by : Greg Berlanti & Andrew Kreisberg Teleplay by : Ben Sokolowski & Brooke Eikmeier December 2 , 2014 ( 2014 - 12 -
02 ) 3J5358 4.34 Eddie proposes a task force to hunt down the Flash . A new metahuman , Roy Bivolo , shows up with the ability to send people into an
uncontrollable rage , and uses his ability to rob a bank . Oliver Queen informs Barry that he and his team are tracking a killer who uses lethal boomerangs .
Barry and Oliver agree to team - up and catch each other 's targets . Barry decides to go after Bivolo by himself . Bivolo uses his abilities on Barry ; but
because of Barry 's powers it makes the effects last longer . Oliver tries to stop Barry , and the two engage in a fight until Wells and Joe use colored strobe
light to reset Barry 's emotional state . Afterward , Barry and Oliver capture Bivolo and place him in the prison at S.T.A.R. Labs .
</item 6>

<item 7>
List of The Flash episodes
Eiling is freed . Iris reveals her feelings for Barry , but states her choice to be with Eddie . Grodd is revealed to be safe and roaming the streets .
Meanwhile , Thawne works on a device that will allow him to return to his own time . He reveals to Eddie , Iris and Barry 's marriage in the future . Later
Thawne completes the device and is revealed to be hiding in a secret chamber inside S.T.A.R. Labs . 22 22 `` Rogue Air '' Doug Aarniokoski Aaron Helbing & Todd
Helbing May 12 , 2015 ( 2015 - 05 - 12 ) 3J5372 3.65 Cisco discovers that Eobard has been supercharging his speed through his wheelchair , which is why he is so
much faster than Barry . The particle accelerator is activated by Eobard , who has managed to repair it . The team is able to find and rescue Eddie , who ends
his relationship with Iris .
</item 7>

<item 8>
List of The Flash episodes
List of the Flash episodes - wikipedia List of the Flash episodes Jump to : navigation , search For the episodes from the 1990 series , see The Flash ( 1990 TV
series ) § Episodes . The Flash is an American action television series developed by Greg Berlanti , Andrew Kreisberg , and Geoff Johns , airing on The CW . It
is based on the DC Comics character Barry Allen / Flash , a costumed crimefighter with the power to move at superhuman speeds . It is a spin - off from Arrow ,
existing in the same fictional universe . The series follows Barry Allen , portrayed by Grant Gustin , a crime scene investigator who gains superhuman speed ,
which he uses to fight criminals , including others who have also gained superhuman abilities . The Flash was renewed in March 2016 for a third season , which
premiered on October 4 , 2016 . On January 8 , 2017 , The CW renewed the series for a fourth season , which premiered on October 10 , 2017 . As of April 24 ,
2018 , 88 episodes of The Flash have aired . Contents ( hide ) 1 Series overview 2 Episodes 2.
</item 8>

<item 9>
List of The Flash episodes
13406 2.46 76 7 `` Therefore I Am '' David McWhirter Eric Wallace & Thomas Pound November 21 , 2017 ( 2017 - 11 - 21 ) T27. 13407 2.20 77 8 `` Crisis on Earth -
X , Part 3 '' Dermott Downs Story by : Andrew Kreisberg & Marc Guggenheim Teleplay by : Todd Helbing November 28 , 2017 ( 2017 - 11 - 28 ) T27. 13408 2.82 78 9
`` Do n't Run '' Stefan Pleszczynski Sam Chalsen & Judalina Neira December 5 , 2017 ( 2017 - 12 - 05 ) T27. 13409 2.22 79 10 `` The Trial of The Flash '' Philip
Chipera Lauren Certo & Kristen Kim January 16 , 2018 ( 2018 - 01 - 16 ) T27.
</item 9>

<item 10>
List of The Flash episodes
25 23 October 4 , 2016 ( 2016 - 10 - 04 ) May 23 , 2017 ( 2017 - 05 - 23 ) 120 3.50 23 October 10 , 2017 ( 2017 - 10 - 10 ) May 22 , 2018 ( 2018 - 05 - 22 ) TBA
TBA Episodes ( edit ) Season 1 ( 2014 -- 15 ) ( edit ) No . overall No. in season Title Directed by Written by Original air date Prod . code U.S. viewers (
millions ) `` Pilot '' David Nutter Story by : Greg Berlanti & Andrew Kreisberg & Geoff Johns Teleplay by : Andrew Kreisberg & Geoff Johns October 7 , 2014 (
2014 - 10 - 07 ) 296648 4.83 Barry Allen , a forensic crime scene assistant for Central City Police Department , is struck by lightning during a storm that is
created after a particle accelerator from S.T.A.R. Labs explodes . In a coma for nine months , Barry awakens in S.T.A.R.
</item 10>
=== END ITEM id=c77e248d8033d96e ===
