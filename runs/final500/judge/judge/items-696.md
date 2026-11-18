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

=== ITEM id=120d383f15b69954 ===
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
They are also credited with changing the `` whole business model of popular music . '' The band has received -- and been nominated for -- multiple awards during
their 55 years as a band ; including three Grammy awards ( and 12 nominations ) , the Juno award for International Entertainer of the Year in 1991 , U.K. 's
Jazz FM Awards Album of the Year ( 2017 ) for their album Blue & Lonesome , and NME awards such as best live band and the NME award for best music film , for
their documentary Crossfire Hurricane . Tours ( edit ) Main article : The Rolling Stones concerts The Rolling Stones ' first concert was on 12 July 1962 at the
Marquee Club in London . The most documented of all the band 's concerts was the Altamont Free Concert at the Altamont Speedway in 1969 . For this concert , the
biker gang Hells Angels provided security , which resulted in a fan , Meredith Hunter , being stabbed and beaten to death by the Angels . Part of the tour and
the Altamont concert were documented in Albert and David Maysles ' film Gimme Shelter .
</item 1>

<item 2>
The Rolling Stones
The Rolling Stones - wikipedia The Rolling Stones Jump to : navigation , search `` Rolling Stones '' redirects here . For other uses , see Rolling Stones (
disambiguation ) . For the magazine , see Rolling Stone . The Rolling Stones The Rolling Stones performing at Summerfest in Milwaukee in 2015 . Left to right :
Charlie Watts , Ronnie Wood , Mick Jagger and Keith Richards Background information Origin London , England Genres Rock blues Years active 1962 -- present
Labels Decca London Rolling Stones Virgin ABKCO Interscope Polydor Associated acts The New Barbarians Billy Preston Faces The Pretty Things John Mayall & the
Bluesbreakers Bill Wyman 's Rhythm Kings Website rollingstones.com Members Mick Jagger Keith Richards Charlie Watts Ronnie Wood Past members Brian Jones Ian
Stewart Dick Taylor Bill Wyman Mick Taylor The Rolling Stones are an English rock band formed in London , England in 1962 . The first stable line - up consisted
of Brian Jones ( guitar , harmonica ) , Mick Jagger ( lead vocals ) , Keith Richards ( guitar , backing vocals ) , Bill Wyman ( bass ) , Charlie Watts ( drums )
, and Ian Stewart ( piano ) .
</item 2>

<item 3>
The Rolling Stones
bass guitar , vocals ( 1962 -- present ) Charlie Watts -- drums , percussion ( 1963 -- present ) Ronnie Wood -- lead and rhythm guitar , bass guitar , backing
vocals ( 1975 -- present ) Former members Brian Jones -- rhythm guitar , harmonica , keyboards , piano , backing vocals ( 1962 -- 1969 ; died 1969 ) Ian Stewart
-- keyboards , piano ( 1962 -- 1963 ; touring 1969 , 1975 -- 1976 , 1978 , 1981 -- 1982 ; died 1985 ) Mick Taylor -- lead guitar , bass guitar , backing vocals
( 1969 -- 1974 ; guest 1981 , 2012 -- 2014 ) Bill Wyman -- bass guitar , keyboards , piano , backing vocals ( 1962 -- 1993 ;
</item 3>

<item 4>
The Rolling Stones
The Rolling Stones were inducted into the Rock and Roll Hall of Fame in 1989 and the UK Music Hall of Fame in 2004 . Rolling Stone magazine ranked them fourth
on the `` 100 Greatest Artists of All Time '' list and their estimated album sales are above 250 million . They have released 30 studio albums , 18 live albums
and numerous compilations . Let It Bleed ( 1969 ) was their first of five consecutive No. 1 studio and live albums in the UK . Sticky Fingers ( 1971 ) was the
first of eight consecutive No. 1 studio albums in the US . In 2008 , the band ranked 10th on the Billboard Hot 100 All - Time Top Artists chart . In 2012 , the
band celebrated its 50th anniversary . Contents ( hide ) 1 History 1.1 Early history 1.2 1962 -- 1964 : Building a following 1.3 1965 -- 1967 : Height of fame
1.4 1968 -- 1972 : `` Back to basics '' 1.
</item 4>

<item 5>
The Rolling Stones
By 1963 they were finding their musical stride as well as popularity , and in 1964 two unscientific opinion polls rated them as Britain 's most popular group ,
even outranking the Beatles . The name of the band was changed shortly after their first gig to `` The Rolling Stones '' . The group 's then acting manager
Giorgio Gomelsky secured a Sunday afternoon residency at the Crawdaddy Club in Richmond in February 1963 , which , Gomelsky claimed , triggered an ``
international renaissance for the blues '' . In May 1963 , The Rolling Stones signed Andrew Loog Oldham as their manager ; he was a former publicist who had
been directed to the band by his previous clients , the Beatles . Because Oldham was only nineteen and had not reached the age of majority -- he was also
younger than anyone in the band -- he could not get an agent 's licence nor sign any contracts without his mother also signing . By necessity he joined with
booking agent Eric Easton in order to secure record financing and assistance booking venues . Gomelsky , who had no written agreement with the band , was not
consulted .
</item 5>

<item 6>
The Rolling Stones
Over time , Jagger has developed into the template for rock frontmen and , with the help of the Stones , has , in the words of the Telegraph , `` changed music
'' through his contributions to it as a pioneer of the modern music industry . Legacy ( edit ) Overhead shot of the Stones concert at Washington -- Grizzly
Stadium in Montana , October 2006 . The Stones have had the highest - grossing concert tour three times Since their formation in 1962 , the Rolling Stones
survived multiple feuds and have gone on to release 30 studio albums , 13 live albums and 109 singles . According to OfficialCharts.com , the Stones are ranked
the fourth bestselling group of all time , with their top single being `` ( I Ca n't Get No ) Satisfaction '' , regarded by many at the time as `` the classic
example of rock and roll . '' The Stones have also contributed to the blues lexicon , creating their own `` codewords '' and slang , which they have used
throughout their catalog of songs , including some of their more popular songs .
</item 6>

<item 7>
The Rolling Stones
Musical development ( edit ) See also : Instruments played by the Rolling Stones A copy of `` Micawber '' , Keith Richards ' signature Telecaster model , in the
Fender Guitar Factory Museum The Rolling Stones have assimilated various musical genres into their own collective sound . Throughout the band 's career , their
musical contributions have been marked by a continual reference and reliance on musical styles including blues , psychedelia , R&B , country , folk , reggae ,
dance , and world music , exemplified by Jones ' collaboration with the Master Musicians of Jajouka , as well as traditional English styles that use stringed
instrumentation like harps . Brian Jones experimented with the use of non-traditional instruments such as the sitar and slide guitar in their early days . The
group started out covering early rock ' n ' roll and blues songs , and have never stopped playing live or recording cover songs . Jagger and Richards shared an
admiration of Jimmy Reed , Muddy Waters , Howlin ' Wolf , and Little Walter , which influenced Brian Jones , of whom Richards says , `` He was more into T -
Bone Walker and jazz blues stuff .
</item 7>

<item 8>
The Rolling Stones
keyboards , piano , backing vocals ( 1962 -- 1993 ; guest 2012 ) Early members Dick Taylor -- bass guitar ( 1962 ) Ricky Fenson -- bass guitar ( 1962 -- 1963 )
Colin Golding -- bass guitar ( 1962 -- 1963 ) Tony Chapman -- drums ( 1962 -- 1963 ) Carlo Little -- drums ( 1962 -- 1963 ) Current touring members Chuck
Leavell -- keyboards , backing vocals ( 1982 -- present ) Bernard Fowler -- backing vocals , percussion ( 1989 -- present ) Darryl Jones -- bass guitar ,
backing vocals ( 1993 -- present ) Matt Clifford -- keyboards , French horn , musical integrator ( 1989 -- 1990 , 2012 -- present ) Tim Ries -- saxophone ,
keyboards ( 1999 -- present ) Karl Denson -- saxophone ( 2014 -- present ) Sasha Allen -- backing vocals , co-lead vocals on `` Gimme Shelter '' ( 2016 --
present ) Former touring musicians Bobby Keys -- saxophone ( 1969 -- 1973 ; 1981 -- 2014 ;
</item 8>

<item 9>
The Rolling Stones
4 1968 -- 1972 : `` Back to basics '' 1.5 1972 -- 1977 : Critical fluctuations and Ronnie Wood 1.6 1978 -- 1982 : Commercial peak 1.7 1983 -- 1988 : Band
turmoil and solo efforts 1.8 1989 -- 1999 : Comeback , return to popularity , and record - breaking tours 1.9 2000 -- 2011 : A Bigger Bang and continued success
1.10 2012 -- present : 50th anniversary and covers album 2 Musical development 3 Legacy 4 Tours 5 Band members 6 Discography 7 See also 8 Notes 9 References 9.1
Footnotes 9.2 Sources 10 Further reading 11 External links History ( edit ) Early History ( edit ) Keith Richards and Mick Jagger became childhood friends and
classmates in 1950 in Dartford , Kent , before the Jagger family moved to Wilmington , five miles ( 8.05 km ) away , in 1954 .
</item 9>

<item 10>
The Rolling Stones
According to Richards , Jones named the band during a phone call to Jazz News . When asked by a journalist for the band 's name , Jones saw a Muddy Waters LP
lying on the floor ; one of the tracks was `` Rollin ' Stone '' . 1962 -- 1964 : building a following ( edit ) The back room of what was the Crawdaddy Club in
Richmond , London where the Rolling Stones had their first residency in 1963 Jones , Jagger , Richards , Stewart and Taylor played a gig billed as `` the Rollin
' Stones '' on 12 July 1962 , at the Marquee Club in London . Shortly afterwards the band went on their first tour of the UK , which they called a `` training
ground '' tour , because it was a new experience for all of them . Their material included the Chicago blues as well as Chuck Berry and Bo Diddley songs . At
that time , the line - up included neither bassist Bill Wyman , who joined in December 1962 , nor drummer Charlie Watts , who joined in January 1963 , thus
completing the band 's original rhythm section .
</item 10>
=== END ITEM id=120d383f15b69954 ===

=== ITEM id=23f5a54d896659d2 ===
Gold sentences and facts:
Sentence 0: Network officials , concerned for the show 's future , turned to Tim Russert , the network 's Washington , D.C. , bureau chief . He took over as moderator of Meet the Press on December 8 , 1991 , and remained with the program until his death on June 13 , 2008 , becoming the longest serving moderator in the program 's history .
  Fact 0.0: Network officials were concerned for the show's future.
  Fact 0.1: Network officials turned to Tim Russert.
  Fact 0.2: Tim Russert was the network's Washington, D.C., bureau chief.
  Fact 0.3: Tim Russert took over as moderator of Meet the Press on December 8, 1991.
  Fact 0.4: Tim Russert remained with Meet the Press until his death on June 13, 2008.
  Fact 0.5: Tim Russert became the longest serving moderator in the history of Meet the Press.
Sentence 1: Martha Rountree 1947 -- 1953 Ned Brooks 1953 -- 1965 Lawrence E. Spivak 1966 -- 1975 Bill Monroe 1975 -- 1984 Roger Mudd and Marvin Kalb ( co-moderators ) 1984 -- 1985 Marvin Kalb 1985 -- 1987 Chris Wallace 1987 -- 1988 Garrick Utley 1989 -- 1991 Tim Russert 1991 -- 2008 Tom Brokaw 2008 David Gregory 2008 -- 2014 Chuck Todd 2014 -- present
  Fact 1.0: Martha Rountree was moderator from 1947 to 1953.
  Fact 1.1: Ned Brooks was moderator from 1953 to 1965.
  Fact 1.2: Lawrence E. Spivak was moderator from 1966 to 1975.
  Fact 1.3: Bill Monroe was moderator from 1975 to 1984.
  Fact 1.4: Roger Mudd and Marvin Kalb were co-moderators from 1984 to 1985.
  Fact 1.5: Marvin Kalb was moderator from 1985 to 1987.
  Fact 1.6: Chris Wallace was moderator from 1987 to 1988.
  Fact 1.7: Garrick Utley was moderator from 1989 to 1991.
  Fact 1.8: Tim Russert was moderator from 1991 to 2008.
  Fact 1.9: Tom Brokaw was moderator in 2008.
  Fact 1.10: David Gregory was moderator from 2008 to 2014.
  Fact 1.11: Chuck Todd has been moderator since 2014.

Retrieved items:
<item 1>
Meet the Press
studios ) 3.2 Notable guests and events 4 Distribution 5 See also 6 References and footnotes 7 External links Format ( edit ) The program 's format consists of
an extended one - on - one interview with the host , and is sometimes followed by a roundtable discussion or one - on - two interview with figures in
adversarial positions , either Congressional members from opposite sides of the aisle or political commentators . Originally a half - hour program for most of
its history , the show expanded to 60 minutes starting with the September 20 , 1992 , broadcast . The program also features in - depth examinations of facts
behind political and general news stories ( particularly as part of a segment called the `` Data Download '' , introduced after Chuck Todd assumed duties as
moderator , which is conducted on a touchscreen within the main set . History ( edit ) Meet the Press set , November 1975 . On this broadcast , a sitting
American president ( Gerald Ford ) was , for the first time , a guest on a live television network news program .
</item 1>

<item 2>
Meet the Press
The next week , Bill Monroe , previously a weekly panelist like Spivak had been years before , took over as moderator and stayed until June 2 , 1984 . For the
next seven and a half years , the program then went through a series of hosts as it struggled in the ratings against ABC 's This Week with David Brinkley .
Roger Mudd and Marvin Kalb , as co-moderators , followed Monroe for a year , followed by Chris Wallace ( who would later to go on to a much longer run as host
of the rival program Fox News Sunday ) from 1987 to 1988 . Garrick Utley , then hosting Weekend Today , concurrently hosted Meet the Press from 1989 through
December 1 , 1991 . All this occurred despite the increasing ratings of NBC News ' other programs ( and those of the network generally ) during that period .
The program originally aired at noon Eastern Time every Sunday , before moving to a 9 : 00 a.m. slot by the early 1990s . Under Russert ( edit ) Meet the Press
logo used from September 10 , 1995 to June 2008 .
</item 2>

<item 3>
Meet the Press
The following week on August 17 , he moderated the entire broadcast . It was also reported on December 1 , 2008 , that the December 7 broadcast would be Brokaw
's last , with Gregory becoming the new permanent host the following Sunday . Under Gregory ( edit ) David Gregory began his tenure as moderator on December 14
, 2008 . Four days after Gregory 's first regular broadcast , on December 18 , NBC News political director Chuck Todd was named contributing editor of Meet the
Press . Throughout Gregory 's tenure as moderator , Meet the Press experienced significant ratings declines . In the final three months of 2013 , the program
placed third among the Sunday morning talk shows in total viewership , behind CBS 's Face the Nation and ABC 's This Week for the first time since 1992 ; it
also experienced the lowest ratings in the show 's entire history among the key 25 - to - 54 age viewing demographic during this period . NBC management became
uncertain as to the future direction of the program .
</item 3>

<item 4>
Meet the Press
This section , in the next two paragraphs , needs additional citations for verification . Please help improve this article by adding citations to reliable
sources . Unsourced material may be challenged and removed . ( September 2014 ) ( Learn how and when to remove this template message ) Meet the Press was
originally presented as a 30 - minute press conference with a single guest and a panel of questioners . Its first guest was James Farley , who served as
Postmaster General , Democratic National Committee chairman and campaign manager to Franklin Delano Roosevelt under the first two terms of the New Deal
Administration . Creator Rountree served as its first host , the program 's only female moderator to date . She stepped down on November 1 , 1953 , and was
succeeded by Ned Brooks , who remained as moderator until his retirement on December 26 , 1965 . Spivak became the moderator on January 1 , 1966 , moving up
from his role as a permanent panelist . He retired on November 9 , 1975 , on a special one - hour edition that featured , for the first time , a sitting
president , Gerald Ford , as guest .
</item 4>

<item 5>
Meet the Press
Bureau Chief and was given `` executive oversight '' of Meet the Press . Interim Brokaw era ( edit ) NBC Nightly News anchor Brian Williams acted as moderator
of the first show following the June 15 tribute to Russert , with the same guests and subject matter that Russert was planning for when he died . Following
Russert 's death , Tom Brokaw was named the interim moderator through the 2008 general elections . Brokaw followed Russert 's tradition by signing off with ``
We 'll be back next Sunday because if it 's Sunday , it 's Meet the Press '' ( a sign - off that would continue to be used by his successors as moderator ) . In
September of that year , the show was presented with limited commercial interruption . On August 10 , David Gregory moderated the panel discussion during the
second half - hour of the broadcast , while Brokaw anchored the first half - hour from the site of the Summer Olympics in Beijing . The following week on August
17 , he moderated the entire broadcast .
</item 5>

<item 6>
Meet the Press
Russert interviews General Peter Pace in 2006 . Network officials , concerned for the show 's future , turned to Tim Russert , the network 's Washington , D.C.
, bureau chief . He took over as moderator of Meet the Press on December 8 , 1991 , and remained with the program until his death on June 13 , 2008 , becoming
the longest serving moderator in the program 's history . Under Russert , the program was expanded to one hour and became less of a televised press conference ,
focusing more on Russert 's questions and comments ; Russert also engaged in longer in - depth interviews and hosted panels of experts to discuss the topics
featured in that week 's broadcast . Russert signed off each edition by saying , `` That 's all for today . We 'll be back next week . If it 's Sunday , it 's
Meet the Press . '' During the professional football season , Russert , a native of Buffalo , New York , and an avid fan of the Buffalo Bills , sometimes added
, `` Go Bills ! , '' and occasionally would ask panelists , `` How ' bout those Sabres ? ''
</item 6>

<item 7>
Meet the Press
It originates from NBC 's bureau in Washington , D.C. The longevity of Meet the Press can be illustrated in consideration of the fact that the program debuted
during what was only the second official `` network television season '' for American television . One historical landmark of the program is that it was the
first on which a sitting U.S. President , Gerald Ford , appeared on a live television network news program , which occurred on the November 9 , 1975 broadcast .
The program has been hosted by 12 different moderators to date , beginning with creator Martha Rountree . The show 's moderator since 2014 is Chuck Todd , who
also serves as political director for NBC News . Currently , the hour - long program airs in most markets on Sundays at 9 : 00 a.m. live in the Eastern Time
Zone and on tape delay elsewhere . Meet the Press is also occasionally pre-empted due to network coverage of sports events held outside of the U.S. The program
is also rebroadcast on Sundays at 2 : 00 p.m. , and Mondays at 2 : 00 a.m. and sometimes 4 : 00 a.m.
</item 7>

<item 8>
Meet the Press
On May 2 , 2010 , Meet the Press became the last NBC News program to convert to high definition , and unveiled a new set consisting of large video screens
mostly used to display Washington scenery , satellite interview subjects and moderator and subject talking points , along with graphics produced for the format
. Moderators ( edit ) The following is the list of moderators for Meet the Press : Martha Rountree 1947 -- 1953 Ned Brooks 1953 -- 1965 Lawrence E. Spivak 1966
-- 1975 Bill Monroe 1975 -- 1984 Roger Mudd and Marvin Kalb ( co-moderators ) 1984 -- 1985 Marvin Kalb 1985 -- 1987 Chris Wallace 1987 -- 1988 Garrick Utley
1989 -- 1991 Tim Russert 1991 -- 2008 Tom Brokaw 2008 David Gregory 2008 -- 2014 Chuck Todd 2014 -- present On - remote broadcasts ( outside D.C. studios ) (
edit ) Date Location Event / Reason July 17 , 1988 Atlanta , Georgia 1988 Democratic National Convention August 14 , 1988 New Orleans ,
</item 8>

<item 9>
Meet the Press
Camera setup Multi-camera Running time 30 minutes ( 1947 -- 1992 ) 52 minutes ( 1992 -- present ) Production company ( s ) NBC News Productions Release Original
network NBC Picture format 480i ( 4 : 3 SDTV ) ( 1947 -- 2010 ) 720p ( 4 : 3 SDTV ) ( 1996 -- 2010 ) 1080i ( 16 : 9 HDTV ) ( 2010 -- present ) Original release
November 6 , 1947 ( 1947 - 11 - 06 ) -- present External links Website www.nbcnews.com/meet-the-press Meet the Press is a weekly American television news /
interview program broadcast on NBC . It is the longest - running program in television history , though the current format bears little resemblance to the debut
episode on November 6 , 1947 . Meet the Press specializes in interviews with leaders in Washington , D.C. , across the country and even the world on issues of
politics , economics , foreign policy and other public affairs , along with panel discussions that provide opinions and analysis . It originates from NBC 's
bureau in Washington , D.C.
</item 9>

<item 10>
Meet the Press
Meet the Press - Wikipedia Meet the Press Jump to : navigation , search This article is about the American public affairs program . For the Australian program ,
see Meet the Press ( Australian TV program ) . Meet the Press Genre Public affairs / news analysis program Created by Martha Rountree Lawrence E. Spivak
Directed by Rob Melick Presented by Chuck Todd ( for past moderators , see section ) Narrated by Fred Facey Bert Pence Dennis Haysbert Theme music composer John
Williams Opening theme `` The Pulse of Events '' ( fourth part of The Mission ) Country of origin United States Original language ( s ) English No. of seasons
70 No. of episodes 3,600 + ( weekly ) ( as of 2017 ) Production Executive producer ( s ) John Reiss Location ( s ) NBC News Washington Bureau , Washington ,
D.C.
</item 10>
=== END ITEM id=23f5a54d896659d2 ===

=== ITEM id=b9d443c349b11650 ===
Gold sentences and facts:
Sentence 0: Draft lotteries were conducted again in 1970 ( for those born in 1951 ) and from 1971 to 1975 ( for 1952 to 1956 births ) . The draft numbers issued in 1972 were never used the next year to call for induction into service , because the last call was December 7 and authority to induct expired on June 30 , 1973 .
  Fact 0.0: Draft lotteries were conducted again in 1970 for men born in 1951.
  Fact 0.1: Draft lotteries were conducted from 1971 to 1975 for men born from 1952 to 1956.
  Fact 0.2: The draft numbers issued in 1972 were never used the next year to call men for induction into service.
  Fact 0.3: The last call for induction was on December 7.
  Fact 0.4: Authority to induct expired on June 30, 1973.
Sentence 1: On December 1 , 1969 , the Selective Service System of the United States conducted two lotteries to determine the order of call to military service in the Vietnam War for men born from 1944 to 1950 . These lotteries occurred during a period of conscription from just before World War II to 1973 . It was the first time a lottery system had been used to select men for military service since 1942 .
  Fact 1.0: On December 1, 1969, the Selective Service System of the United States conducted two lotteries.
  Fact 1.1: The 1969 lotteries determined the order of call to military service in the Vietnam War.
  Fact 1.2: The 1969 lotteries were for men born from 1944 to 1950.
  Fact 1.3: The 1969 lotteries occurred during a period of conscription from just before World War II to 1973.
  Fact 1.4: The 1969 lotteries were the first time a lottery system had been used to select men for military service since 1942.

Retrieved items:
<item 1>
Draft lottery (1969)
Draft lottery ( 1969 ) - wikipedia Draft lottery ( 1969 ) Jump to : navigation , search This article includes a list of references , but its sources remain
unclear because it has insufficient inline citations . Please help to improve this article by introducing more precise citations . ( October 2017 ) ( Learn how
and when to remove this template message ) Representative Alexander Pirnie ( R - NY ) drawing the first number . On December 1 , 1969 , the Selective Service
System of the United States conducted two lotteries to determine the order of call to military service in the Vietnam War for men born from 1944 to 1950 . These
lotteries occurred during a period of conscription from just before World War II to 1973 . It was the first time a lottery system had been used to select men
for military service since 1942 . The lottery numbers assigned in December 1969 were used during calendar year 1970 both to call for induction and to call for
physical examination , a preliminary call covering more men .
</item 1>

<item 2>
Draft lottery (1969)
The draft numbers issued in 1972 were never used the next year to call for induction into service , because the last call was December 7 and authority to induct
expired on June 30 , 1973 . The 1972 to 1975 lottery numbers were used to call some men born 1953 to 1956 for physical exams . The highest number called for a
physical was 215 ( for tables 1970 through 1976 ) . Origins and consequences ( edit ) The reason for the 1969 draft was to add personnel to a war to stop the
spread of communism in Vietnam . After World War II , Japan left Vietnamese Emperor Bao Dai in command . Ho Chi Minh , another Vietnamese leader , saw this as
an opportunity to strike and took over Hanoi . Ho Chi Minh then set up the Democratic Republic of Vietnam . Emperor Bo , in the South , set up the Vietnam State
with Saigon as the capital . Ho Chi Minh was basing his political structure off of the communist states such as Russia while Emperor Bao Dai in the south wanted
a Vietnam that was modeled after the west , like the United States .
</item 2>

<item 3>
Draft lottery (1969)
The 1969 draft lottery only encouraged resentment of the Vietnam War and the draft . It strengthened the anti-war movement , and all over the United States ,
people decried discrimination by the draft system `` against low - education , low - income , underprivileged members of society '' . The draft lottery had
social and economic consequences because it generated resistance to military service and the resisters , draft evaders or `` draft dodgers '' , were generally
young , well - educated , healthy men . The fear of military service in Vietnam influenced many young men born in the late 1940s to join the National Guard .
These young men were aware that the National Guard would be unlikely to send its soldiers to Vietnam . Many men were unable to join the National Guard , even
though they had passed their physicals , because many state National Guards had long waiting lists just to enlist . Still others chose legal sanctions such as
imprisonment , either showing their disapproval by burning their draft cards or draft letters , or simply not presenting themselves for the military service
test . Others left the country , commonly moving to Canada .
</item 3>

<item 4>
Draft lottery (1969)
( 1969 ) . `` The Cost of the Draft and the Cost of Ending the Draft '' . American Economic Review . 59 ( 3 ) . Jump up ^ Ifill , Gwen ( 13 February 1992 ) . ``
THE 1992 CAMPAIGN : New Hampshire ; Clinton Thanked Colonel in ' 69 For ' Saving Me From the Draft ' '' -- via NYTimes.com . ^ Jump up to : Rosenblatt , J.R. ;
Filliben , J.J. ( 1971 ) . `` Randomization and the Draft Lottery '' . Science . 171 : 306 -- 08 . doi : 10.1126 / science. 171.3968. 306 . Jump up ^ `` How the
U.S. Draft Works '' . HowStuffWorks. 2001 - 10 - 18 . Retrieved 2017 - 12 - 05 . External links ( edit ) Landscaper.net ( 2009 ) . The Military Draft and 1969
Draft Lottery for the Vietnam War . Last modified 2009 - 03 - 24 .
</item 4>

<item 5>
Draft lottery (1969)
Others left the country , commonly moving to Canada . The number of American citizens who moved to Canada during the Vietnam war because of the draft is
estimated to be around 125,000 ; it is believed that about half returned to the United States after the Nixon era ( when the war was also over ) ( 1975 ) .
Modification ( edit ) Lottery procedure was improved the next year although public discontent continued to grow until `` authority to induct expired on June 30
, 1973 '' . In 1970 , covering 1951 birthdates for use during 1971 ( sometimes called the 1971 draft ) , scientists at the National Bureau of Standards prepared
78 random permutations of the numbers 1 to 366 using random numbers selected from published tables . From the 78 permutations , 25 were selected at random and
transcribed to calendars using 1 = January 1 , 2 = January 2 , ... 365 = December 31 . Those calendars were sealed in envelopes. 25 more permutations were
selected and sealed in 25 more envelopes without transcription to calendars . The two sets of 25 envelopes were furnished to the Selective Service System .
</item 5>

<item 6>
Draft lottery (1969)
Jump up ^ 91st U.S. Congress . `` AN ACT To amend the Military Selective Service Act of 1967 ... '' ( pdf ) . United States Government Printing Office . ( Pub.
L. 91 -- 124 , 83 Stat. 220 , enacted November 26 , 1969 ) Jump up ^ Peters , Gerhard ; Woolley , John T. `` Richard Nixon : `` Executive Order 11497 - Amending
the Selective Service Regulations to Prescribe Random Selection , '' November 26 , 1969 `` . The American Presidency Project . University of California - Santa
Barbara . ^ Jump up to : `` Vietnam War Protests - Vietnam War - HISTORY.com '' . HISTORY.com . Retrieved 2017 - 12 - 05 . Jump up ^
https://www.nytimes.com/1984/09/02/us/college-enrollment-linked-to-vietnam-war.html , https://timesmachine.nytimes.com/timesmachine/1967/03/19/107186611.pdf
Jump up ^ Fisher , Anthony C. ( 1969 ) .
</item 6>

<item 7>
Draft lottery (1969)
Due to the heavy demand for military personnel , the United States required more than what the regular military could provide , causing the acceleration of the
draft . Between 1965 and 1972 the draft provided 2,215,000 service members to the U.S. military . Previously in the United States , during the War of 1812 ,
President Madison established a commission to recommend the best ways to raise military manpower ; to keep the draft or to institute a volunteer army . After
much debate within the Nixon administration and Congress , it was decided that an all - volunteer force was affordable , feasible , and would enhance the nation
's security . President Richard Nixon issued an executive order prescribing regulations for random selection by the United States Selective Service on November
26 , 1969 . In the 1960s anti-war movements started to occur in the U.S. , mainly among those on college campuses and in more leftist circles , especially those
who embraced the `` hippie '' lifestyle . College students were entitled to a deferment ( 2 - S status ) but were subject to the draft if they dropped out or
graduated . In 1967 , the number of U.S.
</item 7>

<item 8>
Draft lottery (1969)
Last modified 2009 - 03 - 24 . Confirmed 2011 - 05 - 26 . -- contemporary news stories , images of Official Orders ( call to physical exam , call to report ) ,
lottery results , Draft Board classifications , Vietnam troop levels , induction statistics 1917 -- 73 . David Lane ( 2003 ) . Introduction to Graphs : Clearing
Up the Draft with Graphs . Connexions. 18 July 2003 . Confirmed 2011 - 05 - 26 . -- list of draft ranks , additional analysis . Selective Service System ( 2009
) . The Vietnam Lotteries . SSS : History and Records . Last updated 2015 - 09 - 19 . Norton Starr ( 1997 ) . `` Nonrandom Risk : The 1970 Draft Lottery '' .
Journal of Statistics Education 5.2 ( 1997 ) . Confirmed 2011 - 05 - 26 . -- The online edition includes instructions for getting the data online and a lesson
plan for statistics class using the 1970 and 1971 draft lottery data .
</item 8>

<item 9>
Draft lottery (1969)
In 1967 , the number of U.S. military personnel in Vietnam was around 500,000 . The war was costing the U.S. $25 billion a year , and many of the young men
drafted were being sent to a war they wanted no part of . Martin Luther King Jr. also started to support the anti-war movement on the grounds of it being
immoral and the amount of African Americans that were being killed . November 15 , 1969 marked the largest anti war movement in the history of the United States
. This protest featured many anti-war political speakers and popular singers of the time . Many people at the time saw Richard Nixon as a liar ; when he took
office , he claimed that he would begin troop withdrawals from Vietnam immediately . After ten months of being in office , the president had yet to start
withdrawals , anyone and the people felt lied to . Later , president Nixon claimed to have been watching sports as the anti-war demonstration took place outside
the White House . The 1970s were a time of turmoil in the United States , beginning with the civil rights movement which set the standards for practices by the
anti-war movement .
</item 9>

<item 10>
Gerald Ford
President Thieu resigned on April 21 , 1975 , publicly blaming the lack of support from the United States for the fall of his country . Two days later , on
April 23 , Ford gave a speech at Tulane University . In that speech , he announced that the Vietnam War was over `` ... as far as America is concerned '' . The
announcement was met with thunderous applause . South Vietnamese refugees arrive on a U.S. Navy vessel during Operation Frequent Wind 1,373 U.S. citizens and
5,595 Vietnamese and third country nationals were evacuated from the South Vietnamese capital of Saigon during Operation Frequent Wind . In that operation ,
military and Air America helicopters took evacuees to U.S. Navy ships off - shore during an approximately 24 - hour period on April 29 to 30 , 1975 ,
immediately preceding the fall of Saigon . During the operation , so many South Vietnamese helicopters landed on the vessels taking the evacuees that some were
pushed overboard to make room for more people .
</item 10>
=== END ITEM id=b9d443c349b11650 ===

=== ITEM id=e068f1336987f881 ===
Gold sentences and facts:
Sentence 0: Downton Abbey is a British period drama television series created by Julian Fellowes and co-produced by Carnival Films and Masterpiece . It first aired on ITV in the United Kingdom on 26 September 2010 , and on PBS in the United States on 9 January 2011 , as part of the Masterpiece Classic anthology . Six series have been made , the sixth airing in the autumn of 2015 in the UK and Ireland , and in January 2016 in the United States . On 26 March 2015 , the sixth series was confirmed to be the final series , with the final episode airing in the UK on 25 December 2015 on ITV . During the course of the programme , 52 episodes of Downton Abbey aired over six series .
  Fact 0.0: Downton Abbey is a British period drama television series.
  Fact 0.1: Downton Abbey was created by Julian Fellowes.
  Fact 0.2: Downton Abbey was co-produced by Carnival Films and Masterpiece.
  Fact 0.3: Downton Abbey first aired on ITV in the United Kingdom on 26 September 2010.
  Fact 0.4: Downton Abbey first aired on PBS in the United States on 9 January 2011.
  Fact 0.5: The PBS airing was part of the Masterpiece Classic anthology.
  Fact 0.6: Six series of Downton Abbey have been made.
  Fact 0.7: The sixth series aired in the autumn of 2015 in the UK and Ireland.
  Fact 0.8: The sixth series aired in January 2016 in the United States.
  Fact 0.9: On 26 March 2015, the sixth series was confirmed to be the final series.
  Fact 0.10: The final episode of Downton Abbey aired in the UK on 25 December 2015 on ITV.
  Fact 0.11: 52 episodes of Downton Abbey aired over six series.

Retrieved items:
<item 1>
List of Downton Abbey episodes
Series 6 No . overall No. in series Title Directed by Written by Original air date UK viewers ( millions ) 44 `` Episode One '' Minkie Spiro Julian Fellowes 20
September 2015 ( 2015 - 09 - 20 ) 10.32 April 1925 . Big changes are afoot . Robert anticipates having to cut costs and reduce staff , worrying Carson . Mrs
Hughes , nervous she will be embarrassed when performing her wifely duties after marrying Carson , asks Mrs Patmore to sound him out ; after much embarrassment
, Carson says he expects them to have a full marriage , which reassures Mrs Hughes . The local hospital is threatened with a take - over by the larger County
Hospital ; the board is split , with Violet objecting , and Isobel approving in the name of efficiency . Rita Bevan , a former chambermaid at the hotel where
Mary and Lord Gillingham stayed , attempts to blackmail Mary by exposing their tryst .
</item 1>

<item 2>
List of Downton Abbey episodes
Molesley and Baxter take a photo of Bates to different pubs in York , and find the one where he had lunch on the day in question , proving he was not in London
when Green was killed . The witness who identified Anna becomes uncertain , and she is released on bail . A big Christmas party is held for the Downton
household . It also serves as a goodbye party for Tom , who has found work as a writer in America and will be taking his daughter Sybbie with him . Bates
returns and enjoys a tender reunion with Anna . During a private break , Violet confesses to Isobel that she and Prince Kuragan had attempted to run away
together . Series 6 ( 2015 ) ( edit ) The sixth series premiered on 20 September 2015 in the UK and on 3 January 2016 in the United States . This series was the
last , the 2015 Christmas special titled `` The Finale '' . Due to increasingly popular ratings worldwide , a film is said to be in production . Series 6 No .
overall No.
</item 2>

<item 3>
List of Downton Abbey episodes
Michael Gregson is about to depart for Germany ; if he becomes a German citizen , he may be able to divorce his insane wife . Lady Rosamund discovers that Edith
spent the night with Gregson , and warns her about damaging her reputation . Lord Gillingham follows Mary back from London and proposes marriage . She refuses ,
saying she is still grieving Matthew 's death . 30 5 `` Episode Five '' Philip John Julian Fellowes 20 October 2013 ( 2013 - 10 - 20 ) 11.39 May 1922 . Alfred
applies to a chef training program at The Ritz , under Escoffier , but is turned down . Carson considers re-hiring Mr Molesley if Alfred leaves , but Molesley ,
unhappy at being demoted to a footman , dithers too long over the offer , and loses the opportunity . Thomas pressures Cora 's new lady 's maid , Baxter , to
spy on everyone for him . When a tenant farmer dies owing significant back rent , Mary and Tom want to end the lease and run the farm themselves .
</item 3>

<item 4>
List of Downton Abbey episodes
List of Downton Abbey episodes - Wikipedia List of Downton Abbey episodes Jump to : navigation , search This article 's plot summaries may be too long or
excessively detailed . Please help improve them by removing unnecessary details and making them more concise . ( January 2016 ) ( Learn how and when to remove
this template message ) Downton Abbey is a British period drama television series created by Julian Fellowes and co-produced by Carnival Films and Masterpiece .
It first aired on ITV in the United Kingdom on 26 September 2010 , and on PBS in the United States on 9 January 2011 , as part of the Masterpiece Classic
anthology . Six series have been made , the sixth airing in the autumn of 2015 in the UK and Ireland , and in January 2016 in the United States . On 26 March
2015 , the sixth series was confirmed to be the final series , with the final episode airing in the UK on 25 December 2015 on ITV . During the course of the
programme , 52 episodes of Downton Abbey aired over six series .
</item 4>

<item 5>
List of Downton Abbey episodes
overall No. in series Title Directed by Written by Original air date UK viewers ( millions ) 26 `` Episode One '' David Evans Julian Fellowes 22 September 2013
( 2013 - 09 - 22 ) 11.96 February 1922 . Six months on , a distraught Lady Mary mourns Matthew , and feels little motherly love for baby George , now six months
old . Lord Grantham and Branson dispute the management of the estate and struggle to pay death duties . Miss O'Brien suddenly leaves to serve Lady Flintshire ,
on her way to India . Needing a maid in a hurry , Cora hires former maid Edna , who has trained as a lady 's maid , on the strength of the reference which Tom
persuaded Mrs Hughes to give her ; Carson and Mrs Hughes see trouble ahead , and warn Tom to keep his distance . Miss West , the nanny of George and Sybbie ,
Mary 's son and Tom 's daughter , respectively , acts insolently with Thomas , and the underbutler lets Cora know of her tendency to leave them to their own
devices .
</item 5>

<item 6>
List of Downton Abbey episodes
With the future of Downton hanging in the balance , Mary and the Dowager Countess conspire to show Downton at its grandest to persuade Mrs Levinson to give
Downton more money and save them from financial ruin . Their attempts are impeded by the oven breaking down , giving Mrs Levinson a chance to take over running
the party herself in radical fashion . She reveals , when finally asked for help , that though she will `` entertain them in New York '' she can not provide
them with money , as her own late husband tied it up in the estate , much as Violet 's did . She tells them they have to adapt to the changing world , and
states her intent to return to America . Meanwhile downstairs , O'Brien and Thomas wage war against one another due to Alfred 's recent promotion to valet to
Matthew , whilst Mrs Levinson 's maid takes a shine to Alfred . Mrs Hughes faces a cancer scare that she shares only with Mrs Patmore and Doctor Clarkson , and
Isobel Crawley finds that Ethel has become a prostitute .
</item 6>

<item 7>
List of Downton Abbey episodes
Christmas special No . overall Title Directed by Written by Original air date UK viewers ( millions ) 52 `` The Finale '' Michael Engler Julian Fellowes 25
December 2015 ( 2015 - 12 - 25 ) 11.22 29 September -- 31 December 1925 : Carson fumbles serving dinner when his hands begin shaking , suffering from an
unspecified hereditary disease his family called ' the palsy ' . Thomas secures a job with an elderly couple with a staff of only three , but is unhappy there .
Edith offers to expand Spratt 's magazine column to a whole page ; Denker discovers Spratt 's secret , and tries to get him dismissed by revealing it to the
Dowager Countess , but she finds the column jolly amusing and keeps him on . Molesley leaves service when the Headmaster offers him a teaching position , plus a
cottage ; upon parting , he and Baxter finally express their mutual romantic affection . Lord Merton tells Isobel he suffers from pernicious anaemia , but still
wants to marry her .
</item 7>

<item 8>
List of Downton Abbey episodes
The Dowager Countess gets a feisty new lady 's maid , Denker , who falls out with pompous butler , Spratt ; Isobel enjoys the resulting spats . Cora and Robert
reconcile . At a point - to - point , Mary ( who got a daringly modern bob cut earlier in York ) encounters Gillingham and his ex-fiancée , Mabel Lane Fox , and
Charles Blake , whose plan to reunite Gillingham and Mabel seems to be working . After visiting Mrs Patmore 's cottage , Carson rather nervously asks Mrs Hughes
if she would be interested in investing in a property with him . Robert 's dog , Isis , appears to be unwell . 41 7 `` Episode Seven '' Philip John Julian
Fellowes 2 November 2014 ( 2014 - 11 - 02 ) 10.77 1924 : Rose introduces her new friend Atticus Aldridge and his parents , Lord and Lady Sinderby , to the
family .
</item 8>

<item 9>
List of Downton Abbey episodes
Edith and Bertie are married on New Year 's Eve and leave to return to Brancaster Castle , where young Marigold is waiting with her step - grandmother . Dr
Clarkson informs Lord Merton that he has only a mild form of anaemia . During the reception , Anna suddenly goes into labour and gives birth to a healthy son .
Carson realizes his illness will ultimately prevent him from performing his duties , and tenders his resignation ; Robert reluctantly accepts , and suggests
that Thomas return as Butler ; Carson can retire with a pension , retaining an oversight role , which pleases all three . Violet accepts that Cora has taken
over her prime position within the family , and they reconcile . Notes ( edit ) Jump up ^ 10.245 million on ITV1 , 919,000 on ITV1 HD , and 601,000 on ITV1 + 1
. Jump up ^ 9.824 million on ITV1 , 978,000 on ITV1 HD , and 531,000 on ITV1 + 1 .
</item 9>

<item 10>
List of Downton Abbey episodes
The farmer 's son appeals directly to Lord Grantham , who agrees he can continue to farm . Cora has to persuade a nervous Mrs Patmore to exchange their ice -
box for a refrigerator . Edith visits a London doctor . Tom considers moving to America . Bates continues to pressure Anna to explain her behaviour . He tells
Mrs Hughes he will resign if she refuses to tell him the truth about Anna ; Mrs Hughes tells him that a stranger broke into the servants ' hall and raped Anna .
Bates tells Anna that he loves her more than ever , and they reconcile . When Mrs Hughes tells him she is happy the matter is all over , Bates makes it clear
that it is not -- he suspects Green committed the crime . The Dowager Countess employs a new gardener as a favour to Mrs Crawley but suspects that he stole
items while working inside the house . 31 6 `` Episode Six '' Philip John Julian Fellowes 27 October 2013 ( 2013 - 10 - 27 ) 11.54 June 1922 . Alfred leaves
Downton when he wins a place for the Ritz chef training after another candidate drops out .
</item 10>
=== END ITEM id=e068f1336987f881 ===

=== ITEM id=4fda35ed2add638f ===
Gold sentences and facts:
Sentence 0: The Eagles headed into their bye week with the best record in the NFL ( 8 -- 1 ) , and with Wentz leading the NFL MVP discussion with an NFL - leading 23 touchdowns to 5 interceptions . During their bye , the Eagles signed former Giants offensive tackle Will Beatty and former Saints linebacker Dannell Ellerbe to not only add depth but also players with Super Bowl experiences . Coming off of the bye week , the Philadelphia Eagles entered a Sunday night showdown with the defending NFC East champion Dallas Cowboys . The Eagles came out sluggish and trailed 9 -- 7 at the half . The team exploded in the second half though , rattling off 30 unanswered points to soundly defeat their rival by a final score of 37 -- 9 . The Eagles were rolling on a nine - game winning streak after a huge blowout win over the Chicago Bears . The Eagles traveled to Seattle in Week 13 in a huge Sunday Night showdown against Russell Wilson and the Seattle Seahawks . The Eagles struggled throughout the whole game , blowing many scoring opportunities and allowing Wilson to throw 3 touchdowns . The Eagles lost 24 -- 10 , ending their win streak . However , the Eagles bounced back in Week 14 win over the Los Angeles Rams in a tough 43 -- 35 win . However , Carson Wentz left the game in the third quarter with a knee injury . It was later found out that Wentz tore his ACL , ending his MVP caliber season . Backup Nick Foles would once again take over as starting quarterback . In Foles 's first start against the New York Giants , he would lead them back from a 20 -- 7 deficit and score 4 touchdowns and win the game 34 -- 29 . Foles struggled in the last two games of the season against the Oakland Raiders and the Dallas Cowboys , and threw a touchdown and 2 interceptions in that span . Despite this , the Eagles finished 13 -- 3 and secured the top seed in the NFC . The Eagles opened as the underdogs , the first time in history that a No. 1 seed has opened up the postseason as an underdog . Despite this , Foles would lead the Eagles past the Atlanta Falcons in the Divisional Round 15 -- 10 . In the next game Foles and the Eagles annihilated the Minnesota Vikings 38 -- 7 in the NFC Championship nicknamed the Minneapolis Massacre , mocking the Minneapolis Miracle from their previous playoff victory . Foles had his best game since week 15 and threw for 352 passing yards and 3 touchdowns including an impressive flea flicker touchdown pass . This win marked the franchise 's third Super Bowl appearance and a berth in Super Bowl LII for a 2005 rematch against Tom Brady and the New England Patriots .
  Fact 0.0: The Eagles headed into their bye week with the best record in the NFL, at 8-1.
  Fact 0.1: Carson Wentz was leading the NFL MVP discussion.
  Fact 0.2: Wentz had an NFL-leading 23 touchdowns.
  Fact 0.3: Wentz had 5 interceptions.
  Fact 0.4: During their bye, the Eagles signed former Giants offensive tackle Will Beatty.
  Fact 0.5: During their bye, the Eagles signed former Saints linebacker Dannell Ellerbe.
  Fact 0.6: The signings added depth to the Eagles.
  Fact 0.7: Beatty and Ellerbe have Super Bowl experience.
  Fact 0.8: Coming off the bye week, the Eagles entered a Sunday night showdown with the defending NFC East champion Dallas Cowboys.
  Fact 0.9: The Eagles came out sluggish against the Cowboys and trailed 9-7 at the half.
  Fact 0.10: In the second half, the Eagles scored 30 unanswered points.
  Fact 0.11: The Eagles defeated the Cowboys 37-9.
  Fact 0.12: The Eagles were on a nine-game winning streak.
  Fact 0.13: The nine-game winning streak came after a huge blowout win over the Chicago Bears.
  Fact 0.14: The Eagles traveled to Seattle in Week 13 for a Sunday Night showdown against Russell Wilson and the Seattle Seahawks.
  Fact 0.15: The Eagles struggled throughout that game, blowing many scoring opportunities.
  Fact 0.16: Wilson threw 3 touchdowns in that game.
  Fact 0.17: The Eagles lost to Seattle 24-10.
  Fact 0.18: The loss ended the Eagles' win streak.
  Fact 0.19: The Eagles bounced back in Week 14 with a tough 43-35 win over the Los Angeles Rams.
  Fact 0.20: Carson Wentz left the game in the third quarter with a knee injury.
  Fact 0.21: It was later found that Wentz tore his ACL.
  Fact 0.22: The ACL tear ended Wentz's MVP-caliber season.
  Fact 0.23: Backup Nick Foles would once again take over as starting quarterback.
  Fact 0.24: In Foles's first start against the New York Giants, he led the Eagles back from a 20-7 deficit.
  Fact 0.25: In that game, Foles scored 4 touchdowns.
  Fact 0.26: The Eagles won that game 34-29.
  Fact 0.27: Foles struggled in the last two games of the season against the Oakland Raiders and the Dallas Cowboys.
  Fact 0.28: Across those two games, Foles threw a touchdown and 2 interceptions.
  Fact 0.29: Despite this, the Eagles finished 13-3.
  Fact 0.30: The Eagles secured the top seed in the NFC.
  Fact 0.31: The Eagles opened the postseason as underdogs.
  Fact 0.32: It was the first time in history that a No. 1 seed opened the postseason as an underdog.
  Fact 0.33: Foles led the Eagles past the Atlanta Falcons 15-10 in the Divisional Round.
  Fact 0.34: In the next game, Foles and the Eagles annihilated the Minnesota Vikings 38-7 in the NFC Championship.
  Fact 0.35: That NFC Championship game was nicknamed the Minneapolis Massacre.
  Fact 0.36: The nickname mocked the Minneapolis Miracle from the Eagles' previous playoff victory.
  Fact 0.37: Foles had his best game since week 15 in the NFC Championship.
  Fact 0.38: Foles threw for 352 passing yards in that game.
  Fact 0.39: Foles threw 3 touchdowns in that game, including an impressive flea flicker touchdown pass.
  Fact 0.40: The win marked the franchise's third Super Bowl appearance.
  Fact 0.41: The win gave the Eagles a berth in Super Bowl LII.
  Fact 0.42: Super Bowl LII was a rematch of 2005 against Tom Brady and the New England Patriots.
Sentence 1: Playoff record ( all - time ) : 22 -- 21 ( last appearance after beating New England Patriots in Super Bowl LII )
  Fact 1.0: The Eagles' all-time playoff record is 22-21.
  Fact 1.1: The record is as of their last appearance, after beating the New England Patriots in Super Bowl LII.

Retrieved items:
<item 1>
Super Bowl XXXIX
Head coach : Bill Belichick Coaches : Romeo Crennel Brian Daboll Jeff Davidson Ivan Fears Pepper Johnson Josh McDaniels Eric Mangini Matt Patricia Markus Paul
Dean Pees Dante Scarnecchia Brad Seely Cory Undlin Charlie Weis Mike Woicik New England Patriots Founded in 1960 Formerly the Boston Patriots ( 1960 -- 70 )
Based and headquartered in Foxborough , Massachusetts Franchise Franchise History Hall of Fame Patriot Place Seasons Players Coaches First - round draft picks
Starting quarterbacks Strategy Broadcasters Stadiums Nickerson Field Fenway Park Alumni Stadium Harvard Stadium Foxboro Stadium Gillette Stadium Culture Billy
Sullivan Robert Kraft Jonathan Kraft Pat Patriot Cheerleaders `` I 'm Shipping Up to Boston '' `` Crazy Train '' `` Your Love '' Patriot Reign Family Guy ``
Patriot Games '' `` 3 Acts of God '' `` Gronkowsbees '' `` Stunning and Brave '' ( South Park episode ) Lore Snowplow Game St. Louis Stallions Tuck Rule Game 16
- 0 Spygate 2007 game vs. New York Giants Helmet Catch Butt fumble Deflategate 28 -- 3 Philly Special Rivalries Buffalo Bills Miami Dolphins New York Jets
Baltimore Ravens Denver Broncos Indianapolis
</item 1>

<item 2>
Super Bowl XXXIX
For the Eagles , with the loss , the city of Philadelphia would not see its first championship since the 76ers swept the 1983 NBA Finals until the next time
their city 's teams played for a championship , when the Phillies won the 2008 World Series . The Eagles did not return to the Super Bowl until 2017 , also with
a 13 - 3 record , and defeating the Falcons and Vikings in the playoffs . The Eagles and the Patriots met again in Super Bowl LII , following the 2017 season ,
with the Eagles taking their revenge 41 -- 33 . This is the last time that a team has won back - to - back Super Bowls , with the Patriots losing their second
playoff game a year later and then , for the next eight years , every team either losing their first playoff game or missing them altogether . The streak was
broken in 2014 when the Seahawks defeated the Panthers and ultimately advanced to the Super Bowl , only to lose to a Patriots goal line stand .
</item 2>

<item 3>
Super Bowl XXXIX
( 1960 -- 1969 ) Seasons ( 58 ) 1960 1961 1962 1963 1964 1965 1966 1967 1968 1969 1970 1971 1972 1973 1976 1977 1978 1979 1980 1981 1982 1984 1985 1986 1988
1989 1990 1991 1992 1993 1994 1995 1997 1998 1999 2000 2001 2002 2003 2005 2006 2007 2008 2009 2011 2012 2013 2014 2015 2016 2017 2018 Championship seasons in
bold Philadelphia Eagles Founded in 1933 Based and headquartered in Philadelphia , Pennsylvania Franchise Franchise History Seasons Coaches Quarterbacks Players
Stadiums Baker Bowl Philadelphia Municipal Stadium Connie Mack Stadium Franklin Field Veterans Stadium Lincoln Financial Field Culture Fly , Eagles Fly Swoop
Curse of Billy Penn The Lombardi Curse Invincible Philadelphia Sports Hall of Fame Matt Guokas Sr . Dan Baker Cheerleaders
</item 3>

<item 4>
Philadelphia Eagles
Foles had his best game since week 15 and threw for 352 passing yards and 3 touchdowns including an impressive flea flicker touchdown pass . This win marked the
franchise 's third Super Bowl appearance and a berth in Super Bowl LII for a 2005 rematch against Tom Brady and the New England Patriots . With Foles at the
helm , the Eagles started off the game strong , leading the Patriots 22 -- 12 at halftime . In the second half the Patriots gained momentum and for the first
time took the lead in the 4th quarter 33 -- 32 . The Eagles rallied back and scored an 11 - yard Touchdown to tight end Zach Ertz . Later the Eagles converted
on a 46 - yard field goal by Jake Elliott to lead 41 -- 33 . On the final play of the game with 9 seconds left , Brady threw a hail - mary pass directed toward
tight end Rob Gronkowski which was bobbled by Patrick Robinson before being dropped in the end - zone as time expired , securing the win for the Eagles 41 -- 33
, and ending the 58 - year long championship drought .
</item 4>

<item 5>
Super Bowl XXXIX
The Patriots , who entered the Super Bowl after compiling a 14 -- 2 regular season record , became the first team since the 1997 -- 1998 Denver Broncos to win
consecutive Super Bowls . New England also became the second team after the Dallas Cowboys to win three Super Bowls in four years . The Eagles were making their
second Super Bowl appearance after posting a 13 -- 3 regular season record . The game was close throughout , with the teams battling to a 14 -- 14 tie by the
end of the third quarter . The Patriots then scored 10 points in the 4th quarter with Corey Dillon 's 2 - yard touchdown run and Adam Vinatieri 's 22 - yard
field goal . The Eagles then cut their deficit to 24 -- 21 , with quarterback Donovan McNabb 's 30 - yard touchdown pass to receiver Greg Lewis , with 1 : 48
remaining in the game but could not sustain the comeback . Overall , New England forced four turnovers , while Patriots wide receiver Deion Branch was named
Super Bowl MVP for recording 133 receiving yards and tied the Super Bowl record with 11 catches .
</item 5>

<item 6>
Super Bowl XXXIX
Super Bowl XXXIX - wikipedia Super Bowl XXXIX Jump to : navigation , search Super Bowl XXXIX New England Patriots ( AFC ) Philadelphia Eagles ( NFC ) 24 21
Total NE 0 7 7 10 24 PHI 0 7 7 7 21 Date February 6 , 2005 ( 2005 - 02 - 06 ) Stadium Alltel Stadium , Jacksonville , Florida MVP Deion Branch , Wide Receiver
Favorite Patriots by 7 Referee Terry McAulay Attendance 78,125 Future Hall of Famers Patriots : None Eagles : Brian Dawkins , Terrell Owens Ceremonies National
anthem The combined choirs of the U.S. Military Academy , the U.S. Naval Academy , the U.S. Air Force Academy , and the U.S. Coast Guard Academy , and U.S. Army
Herald Trumpets .
</item 6>

<item 7>
Philadelphia Eagles
1991 1992 1993 1994 1995 1997 1998 1999 2000 2001 2002 2003 2005 2006 2007 2008 2009 2011 2012 2013 2014 2015 2016 2017 2018 Championship seasons in bold Links
to related articles National Football League ( 2018 ) AFC East North South West Buffalo Bills Miami Dolphins New England Patriots New York Jets Baltimore Ravens
Cincinnati Bengals Cleveland Browns Pittsburgh Steelers Houston Texans Indianapolis Colts Jacksonville Jaguars Tennessee Titans Denver Broncos Kansas City
Chiefs Los Angeles Chargers Oakland Raiders NFC East North South West Dallas Cowboys New York Giants Philadelphia Eagles Washington Redskins Chicago Bears
Detroit Lions Green Bay Packers Minnesota Vikings Atlanta Falcons Carolina Panthers New Orleans Saints Tampa Bay Buccaneers Arizona Cardinals Los Angeles Rams
San Francisco 49ers Seattle Seahawks Seasons Seasons ( by team ) Preseason Hall of Fame Game American Bowl Regular season Kickoff game Monday Night Football
International Series London Toronto Bills Series List of games played outside the U.S. Thanksgiving games Christmas games Playoffs Streaks Droughts AFC
Championship NFC Championship Super Bowl champions quarterbacks Pro Bowl History League history Executive history Championship history Timeline Defunct
franchises Franchise
</item 7>

<item 8>
Philadelphia Eagles
Philadelphia Eagles . September 9 , 2016 . Retrieved September 17 , 2017 . Jump up ^ `` Philadelphia Eagles Team Capsule '' ( PDF ) . 2017 Official National
Football League Record and Fact Book . National Football League . August 22 , 2017 . Retrieved September 17 , 2017 . Jump up ^ `` Top 10 NFL Rivalries Of All
Time : No. 4 Giants - Eagles '' . Sports Illustrated . December 15 , 2005 . Retrieved September 17 , 2017 . Jump up ^ Chadiha , Jeffri ( October 31 , 2007 ) .
`` Ranking the NFL 's best rivalries : Where does Colts - Pats fit ? '' . ESPN.com . Retrieved April 12 , 2008 . Jump up ^ Bryan , Dave ( September 20 , 2016 )
. `` After 8 Straight Losses , Steelers Looking For Philadelphia Freedom Sunday Against Eagles '' . SteelersDepot.com . Retrieved September 17 , 2017 .
</item 8>

<item 9>
Doug Pederson
85 Terry Mickens 86 Antonio Freeman 87 Robert Brooks 88 Keith Jackson 89 Mark Chmura 90 Darius Holland 91 Shannon Clavelle 92 Reggie White 93 Gilbert Brown 94
Bob Kuberski 95 Keith McKenzie 96 Sean Jones 98 Gabe Wilkins Head coach : Mike Holmgren Coaches : Larry Brooks Nolan Cromwell Gil Haskell Johnny Holland Sherman
Lewis Jim Lind Tom Lovat Marty Mornhinweg Andy Reid Gary Reynolds Fritz Shurmur Harry Sydney Bob Valesente Philadelphia Eagles Super Bowl LII champions 4 Jake
Elliott 6 Caleb Sturgis 7 Nate Sudfeld 8 Donnie Jones 9 Nick Foles ( MVP ) 10 Mack Hollins 11 Carson Wentz 13 Nelson Agholor 14 Marcus Johnson 16 Bryce Treggs
17 Alshon Jeffery 18 Shelton Gibson 19 Rashard Davis 21 Patrick Robinson 22 Sidney Jones 23 Rodney McLeod 24 Corey Graham 26 Jaylen Watkins 27 Malcolm Jenkins
28 Wendell Smallwood 29 LeGarrette Blount 30 Corey Clement 31 Jalen Mills 32 Rasul Douglas 33 De'Vante Bausby 34 Donnel
</item 9>

<item 10>
Philadelphia Eagles
Season records Main article : List of Philadelphia Eagles seasons Regular season record ( all - time ) : 568 -- 594 -- 26 Playoff record ( all - time ) : 22 --
21 ( last appearance after beating New England Patriots in Super Bowl LII ) Most points in a season : 474 points ( 2014 ) NFL championships won : 4 ( 3 before
the 1967 NFL - AFL merger created the Super Bowl ) Super Bowls won : 1 out of 3 appearances Passing leader ( all - time ) : Donovan McNabb -- 32,873 yards
Rushing leader ( all - time ) : LeSean McCoy -- 6,792 yards Receiving leader ( all - time ) : Harold Carmichael -- 8,978 yards Winningest coach ( all - time ) :
Andy Reid -- 130 wins Top player by approximate value ( all - time ) : Donovan McNabb -- 126 AV Rivalries New York Giants Main article : Eagles -- Giants
rivalry One of the NFL 's oldest , this rivalry began on October 15 , 1933 when the Giants defeated the newly founded Eagles 56 -- 0 .
</item 10>
=== END ITEM id=4fda35ed2add638f ===

=== ITEM id=75d87e719a899030 ===
Gold sentences and facts:
Sentence 0: Due to the high salinity , very few fish species can tolerate living in the Salton Sea . Introduced tilapia are the main fish that can tolerate the high salinity levels and pollution . Other freshwater fish species live in the rivers and canals that feed the Salton Sea , including threadfin shad , carp , red shiner , channel catfish , white catfish , largemouth bass , mosquitofish , sailfin molly , and the vulnerable desert pupfish .
  Fact 0.0: Due to the high salinity, very few fish species can tolerate living in the Salton Sea.
  Fact 0.1: Introduced tilapia are the main fish that can tolerate the high salinity levels and pollution.
  Fact 0.2: Other freshwater fish species live in the rivers and canals that feed the Salton Sea.
  Fact 0.3: Threadfin shad live in the rivers and canals that feed the Salton Sea.
  Fact 0.4: Carp live in the rivers and canals that feed the Salton Sea.
  Fact 0.5: Red shiner live in the rivers and canals that feed the Salton Sea.
  Fact 0.6: Channel catfish live in the rivers and canals that feed the Salton Sea.
  Fact 0.7: White catfish live in the rivers and canals that feed the Salton Sea.
  Fact 0.8: Largemouth bass live in the rivers and canals that feed the Salton Sea.
  Fact 0.9: Mosquitofish live in the rivers and canals that feed the Salton Sea.
  Fact 0.10: Sailfin molly live in the rivers and canals that feed the Salton Sea.
  Fact 0.11: Desert pupfish live in the rivers and canals that feed the Salton Sea.
  Fact 0.12: Desert pupfish are vulnerable.

Retrieved items:
<item 1>
Salton Sea
A lake exists only during times it is replenished by the rivers and rainfall , a cycle that has repeated itself many times over hundreds of thousands of years ,
perhaps cycling every 400 to 500 years . Evidence that the basin was occupied periodically by multiple lakes includes wave - cut shorelines at various
elevations preserved on the hillsides of the east and west margins of the present lake , the Salton Sea . These indicate that the basin was occupied
intermittently as recently as a few hundred years ago . The last of the Pleistocene lakes to occupy the basin was Lake Cahuilla , also periodically identified
on older maps as Lake LeConte or the Blake Sea , after American professor and geologist William Phipps Blake . Throughout the Spanish period of California 's
history , the area was referred to as the `` Colorado Desert '' after the Colorado River . In a railroad survey completed in 1855 , it was called `` the Valley
of the Ancient Lake '' . On several old maps from the Library of Congress , it has been found labeled `` Cahuilla Valley '' ( after the local Native American
tribe ) and `` Cabazon Valley '' ( after a local Native American chief -- Chief Cabazon ) .
</item 1>

<item 2>
Salton Sea
Ecology ( edit ) See also : Sonny Bono Salton Sea National Wildlife Refuge § Wildlife Fish population ( edit ) Due to the high salinity , very few fish species
can tolerate living in the Salton Sea . Introduced tilapia are the main fish that can tolerate the high salinity levels and pollution . Other freshwater fish
species live in the rivers and canals that feed the Salton Sea , including threadfin shad , carp , red shiner , channel catfish , white catfish , largemouth
bass , mosquitofish , sailfin molly , and the vulnerable desert pupfish . Avian population ( edit ) The Salton Sea has been termed a `` crown jewel of avian
biodiversity '' by Dr. Milt Friend of the Salton Sea Science Office . Over 400 species have been documented at the Salton Sea . The most diverse and probably
the most significant populations of bird life in the continental United States are hosted , rivaled only by Big Bend National Park in Texas . It supports 30 %
of the remaining population of the American white pelican . The Salton Sea is also a major resting stop on the Pacific Flyway .
</item 2>

<item 3>
Salton Sea
The Salton Sea is also a major resting stop on the Pacific Flyway . On 18 November 2006 , a Ross 's gull , a high Arctic bird , was sighted and photographed
there . Increasing salinity ( edit ) Dead fish on the western shore of Salton City The lack of an outflow means the Salton Sea does not have a natural
stabilization system ; it is very dynamic . Fluctuations in the water level caused by variations in agricultural runoff , the ancient salt deposits in the lake
bed , and the relatively high salinity of the inflow feeding the sea are all causing ever - increasing salinity . The body was initially a freshwater lake , but
by the 1960s its rising salinity had begun to jeopardize some of the species in it . With a salinity now exceeding 5.0 % w / v ( saltier than seawater ) , most
species of fish can no longer survive there . A freshwater fish notable for its ability to withstand the rising salinity of the Salton Sea , the desert pupfish
, can survive salinities ranging from 0.0 % to 7.0 % .
</item 3>

<item 4>
Salton Sea
Fertilizer runoffs have resulted in eutrophication , with large algal blooms and elevated bacterial levels . By 2014 , large swaths of lake bed were exposed and
salt levels drastically increased due to mandated water transfers to metropolitan areas along the coast and other factors , limiting the water inflow . Besides
the resulting fish kills , the shrinking lake interrupts the bird migration , causes dust clouds , and impacts local tourism negatively . Remediation efforts (
edit ) Past efforts and proposals for a Sea level canal ( edit ) Map of New River and Alamo River courses Alternatives for `` saving '' the Salton Sea have been
evaluated since 1955 . The New River passes from Mexicali , Baja California , to the Imperial Valley , and on to the Salton Sea . Much of the current interest
in the sea was sparked in the 1990s by Congressman Sonny Bono . His widow , Mary Bono Mack , elected to fill his seat , has continued to be interested in the
Salton Sea , as has Representative Jerry Lewis of Redlands . In 1998 , the Sonny Bono Salton Sea Restoration Project was named for the politician .
</item 4>

<item 5>
Salton Sea
San Diego County Water Authority . Retrieved 2014 - 10 - 14 . Jump up ^ See http://www.usbr.gov/lc/region/programs/SaltonSea_2004-2014.xlsx , linked at
https://www2.usgs.gov/saltonsea/monitoring.html Jump up ^ Khaled M. Bali ( 5 October 2013 ) . `` Salton Sea and Salinity '' . UC Davis , Cooperative Extension
Imperial County . Retrieved 2015 - 06 - 29 . The salinity of the Pacific Ocean is approximately 35,000 mg / l ( 1 ppm = 1 mg / l ) . Colorado River water
salinity is about 650 -- 700 mg / L. Salton Sea salinity is about 44,000 mg / L , that is approximately 4.4 % salt . The amount of salts that is deposited in
the Imperial Valley agricultural land with irrigation water is approximately four million tons of salts annually . Jump up ^ Alles , DL ( 2007 - 08 - 06 ) . ``
Geology of the Salton Trough '' ( PDF ) .
</item 5>

<item 6>
Salton Sea
depth 43 ft ( 13 m ) Water volume 6,000,000 acre ⋅ ft ( 7.4 km ) Surface elevation − 236 ft ( − 71.9 m ) ( below sea level ) Settlements Bombay Beach , Desert
Beach , Desert Shores , Salton City , Salton Sea Beach , North Shore References U.S. Geological Survey Geographic Names Information System : Salton Sea Map of
the Salton Sea drainage area The Salton Sea is a shallow , saline , endorheic rift lake located directly on the San Andreas Fault , predominantly in California
's Imperial and Coachella valleys . The lake occupies the lowest elevations of the Salton Sink in the Colorado Desert of Imperial and Riverside Counties in
Southern California . Its surface is 236.0 ft ( 71.9 m ) below sea level as of January 2018 . The deepest point of the sea is 5 ft ( 1.5 m ) higher than the
lowest point of Death Valley . The sea is fed by the New , Whitewater , and Alamo rivers , as well as agricultural runoff , drainage systems , and creeks .
</item 6>

<item 7>
Salton Sea
Many of the species of fish that lived in the sea have been killed off by the combination of pollutants , salt levels , and algal blooms . Dead fish have been
known to wash up in mass quantities on the beaches . The smell of the lake , combined with the stench of the decaying fish , also contributed to the decline of
the tourist industry around the Salton Sea . Many people now visit the Salton Sea and the surrounding settlements to explore the abandoned structures . The town
of Niland is 1.5 miles ( 3 km ) southeast of the sea , with a population of 1,006 . Evidence of geothermal activity is also visible . Mudpots and mud volcanoes
are found on the eastern side of the Salton Sea . A number of geothermal electricity generation plants are located along the southeastern shore of the Salton
Sea in Imperial County . The US Geological Survey describes the smell as `` objectionable '' , `` noxious '' , `` unique '' , and `` pervasive '' .
</item 7>

<item 8>
Salton Sea
Salton Sea - wikipedia Salton Sea Jump to : navigation , search For other uses , see Salton Sea ( disambiguation ) . Salton Sea Location Colorado Desert
Imperial and Riverside Counties , California , U.S. Coordinates 33 ° 18 ′ N 115 ° 48 ′ W ﻿ / ﻿ 33.3 ° N 115.8 ° W ﻿ / 33.3 ; - 115.8 Coordinates : 33 ° 18 ′ N
115 ° 48 ′ W ﻿ / ﻿ 33.3 ° N 115.8 ° W ﻿ / 33.3 ; - 115.8 Type Endorheic rift lake Primary inflows Alamo River New River Whitewater River Primary outflows None
Catchment area 8,360 square miles ( 21,700 km ) Basin countries United States , Mexico Surface area 343 sq mi ( 889 km ) Max .
</item 8>

<item 9>
Salton Sea
1 Past efforts and proposals for a sea level canal 3.1. 2 State restoration plan 4 Earthquake geology 5 Water temperature 6 In popular culture 7 See also 8
References 9 Further reading 10 External links History ( edit ) The area was once part of a vast inland sea that covered a large area of Southern California .
Geologists estimate that for three million years , at least through all the years of the Pleistocene glacial age , a large delta was deposited by the Colorado
River in the southern region of the Imperial Valley . Eventually , the delta reached the western shore of the Gulf of California , creating a barrier that
separated the area of the Salton Sea from the northern reaches of the Gulf . Were it not for this barrier the entire Salton Sink along with the Imperial Valley
would be submerged as the Gulf would extend as far north as Indio . Since the exclusion of the ocean , the Salton Basin has over the ages been alternately a
freshwater lake , an increasingly saline endorheic lake , and a dry desert basin , depending on river flows and the balance between inflow and evaporative loss
.
</item 9>

<item 10>
Salton Sea
Surface water temperature ( Sandy Beach ) Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec 57 ° F 14 ° C 57 ° F 14 ° C 64 ° F 18 ° C 72 ° F 22 ° C 77 ° F 25 ° C
82 ° F 28 ° C 90 ° F 32 ° C 91 ° F 33 ° C 86 ° F 30 ° C 81 ° F 27 ° C 72 ° F 22 ° C 61 ° F 17 ° C In popular culture ( edit ) An abandoned boat stuck in the
ground , close to the west coast marina of the Salton Sea Films The documentary Plagues & Pleasures on the Salton Sea ( which premiered at the 2004 Slamdance
Film Festival and , with the addition of John Waters as narrator , premiered at the Provincetown International Film Festival in Massachusetts and opened in
select theaters on February 24 , 2006 ) , covers the first 100 years of the Salton Sea , along with the environmental issues and offbeat residents of the region
.
</item 10>
=== END ITEM id=75d87e719a899030 ===
