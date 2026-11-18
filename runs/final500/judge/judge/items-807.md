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

=== ITEM id=a5e433786852db15 ===
Gold sentences and facts:
Sentence 0: `` What a Friend We Have in Jesus '' is a Christian hymn originally written by Joseph M. Scriven as a poem in 1855 to comfort his mother who was living in Ireland while he was in Canada . Scriven originally published the poem anonymously , and only received full credit for it in the 1880s . The tune to the hymn was composed by Charles Crozat Converse in 1868 . William Bolcom composed a setting of the hymn .
  Fact 0.0: "What a Friend We Have in Jesus" is a Christian hymn.
  Fact 0.1: The hymn "What a Friend We Have in Jesus" was originally written by Joseph M. Scriven as a poem in 1855.
  Fact 0.2: Joseph M. Scriven wrote the poem to comfort his mother.
  Fact 0.3: Joseph M. Scriven's mother was living in Ireland while he was in Canada.
  Fact 0.4: Joseph M. Scriven originally published the poem anonymously.
  Fact 0.5: Joseph M. Scriven only received full credit for the poem in the 1880s.
  Fact 0.6: The tune of the hymn "What a Friend We Have in Jesus" was composed by Charles Crozat Converse in 1868.
  Fact 0.7: William Bolcom composed a setting of the hymn "What a Friend We Have in Jesus".

Retrieved items:
<item 1>
Graph query row:
person: Charles Crozat Converse
work: What a Friend We Have in Jesus
role: None
relationship: CREATED
</item 1>

<item 2>
What a Friend We Have in Jesus
and piano score for `` What a Friend We Have in Jesus '' at HymnSite.com `` What a Friend We Have in Jesus '' Mp3 and lyrics ( hide ) Washington Phillips
Original Columbia 78 rpm records ( chronological ) `` Lift Him Up That 's All '' / `` Take Your Burden to the Lord and Leave It There '' `` Denomination Blues
-- Part 1 '' / `` Denomination Blues -- Part 2 '' `` Mother 's Last Word to Her Son '' / `` Paul and Silas in Jail '' `` What Are They Doing in Heaven Today ''
/ `` Jesus Is My Friend '' `` I Am Born to Preach the Gospel '' / `` Train Your Child '' `` A Mother 's Last Word to Her Daughter '' / `` I 've Got the Key to
the Kingdom '' `` I Had a Good Father and Mother '' / `` The Church Needs Good Deacons '' Later song releases `` You Ca n't Stop a Tattler -- Part 1 '' `` You
Ca n't Stop a Tattler -- Part 2 '' Related articles List of songs ( with recording & catalogue details ) Retrieved from `` https://en.wikipedia.org/w/index.
</item 2>

<item 3>
What a Friend We Have in Jesus
The tune was in the labor protest movement due to its wide popularity . `` Dump the Bosses Off Your Backs '' ( words by John Brill ) was published in the IWW ``
Little Red Songbook '' and has been recorded by dozens of artists including Utah Phillips , Anne Feeney , and Joe Glazer . A Norwegian version was used in the
film Trollhunter . Published versions ( edit ) Rise Up Singing , page 98 The Lutheran Hymnal , hymn # 457 References ( edit ) Wikisource has original text
related to this article : What a Friend We Have in Jesus Jump up ^ `` What a Friend We Have in Jesus '' . Retrieved 2007 - 05 - 03 . Jump up ^ Polack , W.G. (
1941 ) . Handbook to the Lutheran Hymnal . St. Louis : Concordia . p. 323 . Jump up ^ `` Washington Phillips discography '' . wirz.de . Retrieved August 24 ,
2015 . Jump up ^ George Washington Phillips : Jesus Is My Friend at AllMusic .
</item 3>

<item 4>
What a Friend We Have in Jesus
wikipedia.org/w/index.php?title=What_a_Friend_We_Have_in_Jesus&oldid=831327264 '' Categories : 1928 songs Canadian Christian hymns Columbia Records singles Glen
Campbell songs Mississippi John Hurt songs Pace Jubilee Singers songs Songs about Jesus Washington Phillips songs Hidden categories : Articles with hAudio
microformats Articles containing Japanese - language text Articles with trivia sections from February 2018 All articles with trivia sections Talk Contents About
Wikipedia Wikisource 日本 語 Norsk Svenska 中文 Edit links This page was last edited on 20 March 2018 , at 00 : 54 . Text is available under the Creative Commons
Attribution - ShareAlike License ; additional terms may apply . By using this site , you agree to the Terms of Use and Privacy Policy . Wikipedia ® is a
registered trademark of the Wikimedia Foundation , Inc. , a non-profit organization . About Wikipedia
</item 4>

<item 5>
What a Friend We Have in Jesus
External links ( edit ) Piano score ( pdf ) at Timeless Truths Text , MIDI , and piano score for `` What a Friend We Have in Jesus '' at HymnSite.
</item 5>

<item 6>
Graph query row:
person: Joseph M. Scriven
work: What a Friend We Have in Jesus
role: None
relationship: CREATED
</item 6>

<item 7>
What a Friend We Have in Jesus
Jump up ^ George Washington Phillips : Jesus Is My Friend at AllMusic . Retrieved August 24 , 2015 . Jump up ^ `` Volkswagen Golf -- Paula Hamilton -- 1987 --
UK Advert '' . Retro Commercials . March 13 , 2010 . Retrieved September 26 , 2011 . Jump up ^ `` Pure Hymns overview '' . Allmusic . Retrieved July 4 , 2010 .
Jump up ^ `` Monty Alexander UPLIFT 2 '' . Jazz Legacy Productions . Retrieved January 11 , 2014 . Jump up ^ `` Itsukushimi Fukaki song sheet '' ( PDF ) .
Retrieved 2007 - 07 - 16 . ^ Jump up to : `` 312 : What a Friend We Have in Jesus '' . Retrieved 2008 - 03 - 19 . Jump up ^ Christhiya Keerthanangal ( Christian
Hymns ) . 2007 . Hymn 173 . External links ( edit ) Piano score ( pdf ) at Timeless Truths Text , MIDI ,
</item 7>

<item 8>
What a Friend We Have in Jesus
What a Friend We Have in Jesus - wikipedia What a Friend We Have in Jesus Jump to : navigation , search `` What a Friend We Have in Jesus '' Song Genre Hymn
Composer ( s ) Charles C. Converse ( 1868 ) Lyricist ( s ) Joseph M. Scriven ( 1855 ) `` What a Friend We Have in Jesus '' is a Christian hymn originally
written by Joseph M. Scriven as a poem in 1855 to comfort his mother who was living in Ireland while he was in Canada . Scriven originally published the poem
anonymously , and only received full credit for it in the 1880s . The tune to the hymn was composed by Charles Crozat Converse in 1868 . William Bolcom composed
a setting of the hymn . The hymn also has many versions with different lyrics in multiple languages . The Handbook to the Lutheran Hymnal notes , `` In spite of
the fact that this hymn , with its tune , has been criticized as being too much on the order of the sentimental gospel type , its popularity remains strong ,
and the hymn retains a place in modern hymnals . ''
</item 8>

<item 9>
What a Friend We Have in Jesus
It is translated as `` Preethi gala mana Yesu '' and it is listed in the Andhra Kraisthava Keerthanalu ( Andhra Christian hymnals ) as hymn No. 407 . In India ,
the hymn sung in Konkani , `` Dulob Jezu '' , was translated by Felix Paul Noronha of Mangalore . In popular culture ( edit ) This section indiscriminately
collects miscellaneous information . Please compress this material to remove any irrelevant or unimportant information . ( February 2018 ) A version was sung
during World War I with the line `` When this bloody war is over '' in place of the original title . The tune was used in the 1969 film , Oh ! What a Lovely War
with the title `` When This Lousy War Is Over '' . The tune was used in Alan Price 's song Changes , which was written for the 1973 film O Lucky Man ! and later
became a hit when used in the 1987 Volkswagen Golf television commercial . The tune was in the labor protest movement due to its wide popularity .
</item 9>

<item 10>
What a Friend We Have in Jesus
In some settings , the lyrics have been matched to other tunes such as the Welsh `` Calon Lân '' ( originally wedded to the Welsh poem translated as `` A Pure
Heart '' ) . What a Friend We Have In Jesus Performed by the Onward Brass Band Problems playing this file ? See media help . Contents ( hide ) 1 Renditions 1.1
In Asia 2 In popular culture 3 Published versions 4 References 5 External links Renditions ( edit ) Washington Phillips , as `` Jesus Is My Friend '' ( 1928 ,
Columbia Records ) Bing Crosby ( 1951 , Beloved Hymns ) Rosemary Clooney ( 1959 , Hymns from the Heart , MGM Records ) Ella Fitzgerald with the Ralph Carmichael
Choir ( 1967 , Brighten the Corner ) Aretha Franklin ( 1972 , Amazing Grace ) Alan Price used the tune for his song `` Changes '' in the 1973 film O Lucky Man !
, which was reused in a Volkswagen commercial of the same name with Paula Hamilton in the 1980s .
</item 10>
=== END ITEM id=a5e433786852db15 ===

=== ITEM id=154882dbdad0b620 ===
Gold sentences and facts:
Sentence 0: Your Lie in April Cover of the first tankōbon volume featuring Kōsei Arima and Kaori Miyazono 四 月 は 君 の 嘘 ( Shigatsu wa Kimi no Uso ) Manga Written by Naoshi Arakawa Published by Kodansha English publisher Kodansha Comics Demographic Shōnen Magazine Monthly Shōnen Magazine Original run April 6 , 2011 -- February 6 , 2015 Volumes 11 ( List of volumes ) Anime television series Directed by Kyōhei Ishiguro Written by Takao Yoshioka Music by Masaru Yokoyama Studio A-1 Pictures Licensed by Madman Entertainment Aniplex of America Anime Limited Original network Fuji TV ( Noitamina ) Original run October 9 , 2014 -- March 19 , 2015 Episodes 22 ( List of episodes ) Light novel Your Lie in April - A Six Person Etude Written by Yui Tokiumi Published by Kodansha English publisher Vertical Published November 17 , 2014 Volumes Manga Shigatsu wa Kimi no Uso : Coda Written by Naoshi Arakawa Demographic Shōnen Original run February 25 , 2015 -- October 28 , 2015 Volumes Original video animation Moments Directed by Kazuya Iwata Written by Takao Yoshioka Music by Masaru Yokoyama Studio A-1 Pictures Released May 15 , 2015 Runtime 23 minutes Live - action film Your Lie in April ( film ) Anime and Manga portal
  Fact 0.0: The cover shows the first tankōbon volume featuring Kōsei Arima and Kaori Miyazono.
  Fact 0.1: The title is 四 月 は 君 の 嘘 (Shigatsu wa Kimi no Uso).
  Fact 0.2: The manga Shigatsu wa Kimi no Uso is written by Naoshi Arakawa.
  Fact 0.3: The manga is published by Kodansha, with English publisher Kodansha Comics.
  Fact 0.4: The manga has the Shōnen demographic and was serialized in Monthly Shōnen Magazine.
  Fact 0.5: The manga's original run was April 6, 2011 to February 6, 2015.
  Fact 0.6: The manga has 11 volumes.
  Fact 0.7: The anime television series is directed by Kyōhei Ishiguro.
  Fact 0.8: The anime television series is written by Takao Yoshioka.
  Fact 0.9: The anime television series has music by Masaru Yokoyama.
  Fact 0.10: The anime television series is by studio A-1 Pictures.
  Fact 0.11: The anime is licensed by Madman Entertainment and Aniplex of America Anime Limited.
  Fact 0.12: The anime's original network is Fuji TV (Noitamina).
  Fact 0.13: The anime's original run was October 9, 2014 to March 19, 2015.
  Fact 0.14: The anime has 22 episodes.
  Fact 0.15: The light novel Your Lie in April - A Six Person Etude is written by Yui Tokiumi.
  Fact 0.16: The light novel is published by Kodansha, with English publisher Vertical.
  Fact 0.17: The light novel was published November 17, 2014.
  Fact 0.18: The manga Shigatsu wa Kimi no Uso: Coda is written by Naoshi Arakawa.
  Fact 0.19: The manga Shigatsu wa Kimi no Uso: Coda has the Shōnen demographic.
  Fact 0.20: The manga Shigatsu wa Kimi no Uso: Coda's original run was February 25, 2015 to October 28, 2015.
  Fact 0.21: The original video animation Moments is directed by Kazuya Iwata.
  Fact 0.22: The original video animation Moments is written by Takao Yoshioka.
  Fact 0.23: The original video animation Moments has music by Masaru Yokoyama.
  Fact 0.24: The original video animation Moments is by studio A-1 Pictures.
  Fact 0.25: The original video animation Moments was released May 15, 2015.
  Fact 0.26: The original video animation Moments has a runtime of 23 minutes.
  Fact 0.27: There is a live-action film titled Your Lie in April.

Retrieved items:
<item 1>
Your Lie in April
2014 -- March 19 , 2015 Episodes 22 ( List of episodes ) Light novel Your Lie in April - A Six Person Etude Written by Yui Tokiumi Published by Kodansha English
publisher Vertical Published November 17 , 2014 Volumes Manga Shigatsu wa Kimi no Uso : Coda Written by Naoshi Arakawa Demographic Shōnen Original run February
25 , 2015 -- October 28 , 2015 Volumes Original video animation Moments Directed by Kazuya Iwata Written by Takao Yoshioka Music by Masaru Yokoyama Studio A-1
Pictures Released May 15 , 2015 Runtime 23 minutes Live - action film Your Lie in April ( film ) Anime and Manga portal Your Lie in April , known in Japan as
Shigatsu wa Kimi no Uso ( 四 月 は 君 の 嘘 ) or just simply Kimiuso , is a Japanese manga series written and illustrated by Naoshi Arakawa . The series was
serialized in Kodansha 's Monthly Shōnen Magazine from April 2011 to May 2015 .
</item 1>

<item 2>
Your Lie in April
wikipedia.org/w/index.php?title=Your_Lie_in_April&oldid=838327454 '' Categories : Manga series 2011 manga 2014 anime television series 2015 manga 2015 anime
OVAs 2015 comics endings 2015 Japanese television series endings A-1 Pictures Comedy - drama anime and manga Kodansha manga Madman Entertainment anime Manga
adapted into films Music - themed anime and manga NoitaminA Romance anime and manga School anime and manga Shōnen manga Slice of life anime and manga Tragedy
anime and manga Works about pianos and pianists Winner of Kodansha Manga Award ( Shōnen ) Hidden categories : CS1 Japanese - language sources ( ja ) Articles
containing Japanese - language text Official website different in Wikidata and Wikipedia Talk Contents About Wikipedia Azərbaycanca Català Español Français 한국어
Bahasa Indonesia Italiano Nederlands 日本 語 Norsk Português Română Русский Simple English Suomi Tagalog ไทย Tiếng Việt 中文 11 more Edit links This page was last
edited on 26 April 2018 , at 09 : 28 .
</item 2>

<item 3>
Your Lie in April
Nē , Mama Kī Te Yo ) 5 January 17 , 2013 ISBN 978 - 4 - 06 - 371359 - 6 December 29 , 2015 ISBN 978 - 1 - 63 - 236175 - 2 `` Falling '' ( 墜ちる , Ochiru ) `` The
Scenery When I 'm with You '' ( 君 と いた 景色 , Kimi Toita Keshiki ) `` Along the Railroad Track '' ( 線路 沿い の 道 , Senro - zoi no Michi ) `` Under the Bridge '' ( 橋
の 下 , Hashi no Shita ) 6 May 17 , 2013 ISBN 978 - 4 - 06 - 371375 - 6 March 29 , 2016 ISBN 978 - 1 - 63 - 236176 - 9 `` Candied Apple '' ( りんご 飴 , Ringo Ame )
`` Twinkle , Twinkle , Little Star '' ( トゥインク ル リトル スタ ,
</item 3>

<item 4>
Your Lie in April
It premiered on October 10 , 2014 in Japan and ended on March 20 , 2015 . The first opening theme song is `` Hikaru Nara '' ( 光る なら , lit . If You Will Shine )
by Goose house and the first ending theme song is `` Kirameki '' ( キラメキ , lit . Sparkle ) by wacci . The second opening song is `` Nanairo Symphony '' ( 七色
シンフォニー , Nanairo Shinfonī , lit . `` Seven Colored Symphony '' ) by Coalamode and the second ending theme is `` Orange '' ( オレンジ , Orenji ) by 7 ! ! ( Seven
Oops ) . The original soundtracks are composed by Masaru Yokoyama . In North America , Aniplex of America licensed the series within the region , who
simulcasted the series on Crunchyroll , Aniplex Channel , Hulu and Viewster . In Australia and New Zealand , the series is licensed by Madman Entertainment ,
who streamed the series on AnimeLab .
</item 4>

<item 5>
Your Lie in April
Kōsei has n't touched in four years ( he was fifteen in the series ) the piano , and views the world in monochrome , without any flair or color , resigning
himself to living out his life with his childhood friends , Tsubaki and Watari , until , one day , a girl changes everything . Kaori Miyazono , an audacious ,
free - spirited , fourteen - year - old violinist whose playing style reflects her manic personality , helps Kōsei return to the music world and shows that it
should be free and mold - breaking unlike the structured and rigid style Kōsei was used to , and as she continues to uplift him , he quickly realizes that he
loves her , though she already likes Watari . However , while performing together ( Kōsei having been dragged into it by Kaori ) , Kaori suddenly collapses
after a moving performance and is hospitalised . At first Kaori says that she is anaemic and just needs some routine testing , but this is revealed to be a lie
when flashes of her past reveal her collapsing and bleeding many times before .
</item 5>

<item 6>
Your Lie in April
Kaori Miyazono ( 宮園 かをり , Miyazono Kawori ) Voiced by : Risa Taneda ( Japanese ) ; Erica Lindbeck ( English ) & Violin performance by : Yūna Shinohara Played by
: Suzu Hirose ( live - action film ) ; Arisa Matsunaga ( stage play ) Kaori is Tsubaki 's classmate ; she is a free - spirited violinist who has drawn much
criticism from judging panels due to her unwillingness to adhere strictly to the score , but is highly favored by audiences that hear her playing . Kaori first
met Kōsei when she asked Tsubaki to set her up with Watari . As their friendship grew , she eventually convinced Kōsei to play the piano again , first as her
accompanist and later in a piano competition . It is revealed at the end of the anime , in a letter addressed to Kōsei , that she had asked to be set up with
Watari in order to meet Kōsei , knowing that Watari and Tsubaki were good friends of his .
</item 6>

<item 7>
Your Lie in April
Your Lie in April - Wikipedia Your Lie in April Jump to : navigation , search This article is about the manga and anime . For the live - action film adaptation
, see Your Lie in April ( film ) . Your Lie in April Cover of the first tankōbon volume featuring Kōsei Arima and Kaori Miyazono 四 月 は 君 の 嘘 ( Shigatsu wa Kimi
no Uso ) Manga Written by Naoshi Arakawa Published by Kodansha English publisher Kodansha Comics Demographic Shōnen Magazine Monthly Shōnen Magazine Original
run April 6 , 2011 -- February 6 , 2015 Volumes 11 ( List of volumes ) Anime television series Directed by Kyōhei Ishiguro Written by Takao Yoshioka Music by
Masaru Yokoyama Studio A-1 Pictures Licensed by Madman Entertainment Aniplex of America Anime Limited Original network Fuji TV ( Noitamina ) Original run
October 9 , 2014 -- March 19 ,
</item 7>

<item 8>
Your Lie in April
Kodansha . Retrieved January 27 , 2016 . Jump up ^ `` Your Lie in April 9 '' . Penguin Random House . Retrieved April 11 , 2016 . Jump up ^ 四 月 は 君 の 嘘 ( 10 ) (
Your Lie in April ( 10 ) ) ( in Japanese ) . Kodansha . Retrieved January 27 , 2016 . Jump up ^ `` Your Lie in April 10 '' . Penguin Random House . Retrieved
November 6 , 2016 . Jump up ^ `` Your Lie in April 11 '' . Penguin Random House . Retrieved April 11 , 2016 . Jump up ^ `` Goose House to Perform Opening ,
Cover Songs for Shigatsu wa Kimi no Uso Anime '' . Anime News Network . August 15 , 2014 . Retrieved August 3 , 2017 . Jump up ^ `` Coala Mode , 7 ! ! Perform
Theme Songs For Your Lie in April Anime 's 2nd Half '' . Anime News Network .
</item 8>

<item 9>
Your Lie in April
The series is licensed in the United Kingdom and Ireland by Anime Limited . The series was later made available on Netflix with the English dub , as well as
English subtitles . Live - action film ( edit ) Main article : Your Lie in April ( film ) On August 24 , 2015 , the domain ' kimiuso-movie.jp ' was registered
under Toho , a Japanese film production and distribution company , which made fans believe that a movie adaptation was in the works . Speculations were
confirmed in September 2015 when the main cast was announced for the live - action adaptation of the series : Kento Yamazaki as Kōsei Arima , Suzu Hirose as
Kaori Miyazono , E-girls ' Anna Ishii as Tsubaki Sawabe , and Taishi Nakagawa as Ryōta Watari . While the original story depicts the characters in their junior
high year , it is announced that they will be in their second - year of high school in the film .
</item 9>

<item 10>
Your Lie in April
An anime television series adaptation by A-1 Pictures aired from October 2014 to March 2015 on Fuji TV 's Noitamina block . A live - action film adaptation of
the same name was released in September 2016 . Contents ( hide ) 1 Plot 2 Characters 3 Media 3.1 Manga 3.1. 1 Volume list 3.2 Anime 3.3 Live - action film 3.4
Stage play 4 Reception 5 References 6 External links Plot ( edit ) Piano prodigy Kōsei Arima dominated various music competitions and has become famous among
child musicians , but also controversial . After the death of his abusive mother , who was also his coldhearted instructor who forced him to play the piano
emotionlessly , he had a mental breakdown at the age of eleven while performing at a piano recital which resulted in him no longer being able to hear the sound
of his piano even though his hearing was perfectly fine .
</item 10>
=== END ITEM id=154882dbdad0b620 ===

=== ITEM id=e65db654cfebf737 ===
Gold sentences and facts:
Sentence 0: The official music video for the song was directed by P.R. Brown and produced by Steve Lamar . The video was filmed along the Californian coast at The Inn at Newport Ranch , a resort and cattle ranch to the north of San Francisco . It features Brown and his duet partner Lauren Alaina singing along the coast . It was released on May 14 , 2017 . An earlier lyric video was first released on May 4 , 2017 , and shows Brown and Alaina cruising down a road .
  Fact 0.0: The official music video for the song was directed by P.R. Brown.
  Fact 0.1: The official music video for the song was produced by Steve Lamar.
  Fact 0.2: The video was filmed along the Californian coast at The Inn at Newport Ranch.
  Fact 0.3: The Inn at Newport Ranch is a resort and cattle ranch.
  Fact 0.4: The Inn at Newport Ranch is north of San Francisco.
  Fact 0.5: The video features Brown and his duet partner Lauren Alaina singing along the coast.
  Fact 0.6: The video was released on May 14, 2017.
  Fact 0.7: An earlier lyric video was first released on May 4, 2017.
  Fact 0.8: The earlier lyric video shows Brown and Alaina cruising down a road.

Retrieved items:
<item 1>
What Ifs
wikipedia.org/w/index.php?title=What_Ifs&oldid=838232791 '' Categories : 2016 songs 2017 singles Kane Brown songs Lauren Alaina songs RCA Records Nashville
singles Vocal duets Song recordings produced by Dann Huff Songs written by Jordan Schmidt Billboard Hot Country Songs number - one singles Hidden categories :
Music infoboxes with deprecated parameters Articles with hAudio microformats Singlechart usages for Canada Singlechart called without song Singlechart usages
for Billboardcanadacountry Singlechart usages for Billboardhot100 Singlechart usages for Billboardcountryairplay Singlechart usages for Billboardcountrysongs
Certification Table Entry usages for Canada Certification Table Entry usages for United States Talk Contents About Wikipedia Español Edit links This page was
last edited on 25 April 2018 , at 18 : 59 . About Wikipedia
</item 1>

<item 2>
What Ifs
Music Canada . Retrieved February 8 , 2018 . Preceded by `` Body Like a Back Road '' by Sam Hunt Billboard Hot Country Songs number - one single October 21 --
November 11 , 2017 Succeeded by `` When It Rains It Pours '' by Luke Combs Preceded by `` All the Pretty Girls '' by Kenny Chesney Billboard Country Airplay
number - one single October 28 , 2017 Kane Brown Studio albums Kane Brown ( 2016 ) EPs Closer ( 2015 ) Chapter 1 ( 2016 ) Singles `` Do n't Go City on Me '' ``
Used to Love You Sober '' `` What Ifs '' `` Heaven '' Lauren Alaina Studio albums Wildflower Road Less Traveled EPs Lauren Alaina Singles `` Like My Mother Does
'' `` Georgia Peaches '' `` Eighteen Inches '' `` Barefoot and Buckwild '' `` Next Boyfriend '' `` Road Less Traveled '' `` Doin ' Fine '' Other singles `` My
Grown Up Christmas List '' `` What Ifs '' Retrieved from `` https://en.wikipedia.org/w/index.
</item 2>

<item 3>
“What Ifs” is a country duet by Kane Brown and Lauren Alaina, released in 2017 as the third single from Brown’s self-titled debut album. Written by Brown,
Matthew McGinn, and Jordan Schmidt, it explores uncertainties in a relationship. The former school friends recorded the duet after Brown requested Alaina’s
participation. The song topped Billboard’s Hot Country Songs chart in October 2017, ending Sam Hunt’s 34-week run, and also reached No.
</item 3>

<item 4>
What Ifs
Jump up ^ Bjorke , Matt ( January 3 , 2018 ) . `` Top 30 Country Digital Singles Chart : January 3 , 2018 '' . Roughstock . Retrieved January 14 , 2017 . ^ Jump
up to : Bjorke , Matt ( March 6 , 2018 ) . `` The Top 30 Digital Country Singles : March 5 , 2018 '' . Roughstock . Retrieved March 11 , 2018 . Jump up ^ ``
Watch Kane Brown , Lauren Alaina 's Picturesque ' What Ifs ' Video '' . Rolling Stone . May 15 , 2017 . Jump up ^ Kimble , Lindsay ( May 4 , 2017 ) . `` From
Middle School Chorus to Nashville , Kane Brown and Lauren Alaina Team Up for Duet ' What Ifs ' '' . Jump up ^ `` Kane Brown Chart History ( Canadian Hot 100 )
'' . Billboard . Retrieved October 24 , 2017 . Jump up ^ `` Kane Brown Chart History ( Canada Country ) '' . Billboard .
</item 4>

<item 5>
What Ifs
Jump up ^ `` Kane Brown Chart History ( Canada Country ) '' . Billboard . Retrieved November 14 , 2017 . Jump up ^ `` Kane Brown Chart History ( Hot 100 ) '' .
Billboard . Retrieved October 24 , 2017 . Jump up ^ `` Kane Brown Chart History ( Country Airplay ) '' . Billboard . Retrieved October 17 , 2017 . Jump up ^ ``
Kane Brown Chart History ( Hot Country Songs ) '' . Billboard . Retrieved October 10 , 2017 . Jump up ^ `` Country Airplay Songs -- Year - End 2017 '' .
Billboard . Retrieved December 18 , 2017 . Jump up ^ `` Hot Country Songs -- Year - End 2017 '' . Billboard . Retrieved December 18 , 2017 . Jump up ^ `` Hot
100 Songs -- Year - End 2017 '' . Billboard . Retrieved December 12 , 2017 . Jump up ^ `` Canadian single certifications -- Kane Brown -- What Ifs '' . Music
Canada .
</item 5>

<item 6>
What Ifs
According to Brown , another song was going to be used in the album , however , he was asked so much about `` What Ifs '' on social media that it was then
decided that the song should be included in the album . Composition ( edit ) The song is in the key of G minor with a moderately fast tempo of approximately 126
beats per minute . It follows the chord progression Gm - B ♭ - F. Brown 's vocals range from F - B ♭ . Commercial performance ( edit ) The song sold 6,000
downloads in the US the week the album was released on December 2 , 2016 , allowing it to chart at number 37 on the US Billboard Hot Country Songs chart . The
song first entered the US Billboard Country Airplay chart for the week of March 4 , 2017 at number 60 . It reached No. 1 on Hot Country Songs in October 2017 ,
ending the record breaking 34 - week reign of Sam Hunt 's `` Body Like a Back Road '' . The song is the first No. 1 on the chart for both Brown and Alaina .
</item 6>

<item 7>
What Ifs
What Ifs - wikipedia What Ifs Jump to : navigation , search `` What Ifs '' Single by Kane Brown featuring Lauren Alaina from the album Kane Brown Released
February 6 , 2017 ( 2017 - 02 - 06 ) Format Digital download Genre Country Length 3 : 08 Label RCA Nashville Songwriter ( s ) Kane Brown Matthew McGinn Jordan
Schmidt Producer ( s ) Dann Huff Kane Brown singles chronology `` Thunder in the Rain '' ( 2016 ) `` What Ifs '' ( 2017 ) `` Heaven '' ( 2017 ) `` Thunder in
the Rain '' ( 2016 ) `` What Ifs '' ( 2017 ) `` Heaven '' ( 2017 ) Lauren Alaina singles chronology `` Road Less Traveled '' ( 2016 ) Road Less Traveled 2016 ``
What Ifs '' ( 2017 ) What Ifs2017 `` Doin ' Fine '' ( 2017 ) Doin ' Fine 2017 `` What Ifs '' is a song recorded by American country music singer Kane Brown
featuring Lauren Alaina for his self - titled debut album .
</item 7>

<item 8>
What Ifs
The song is the first No. 1 on the chart for both Brown and Alaina . The following week , for charts dated October 28 , 2017 , `` What Ifs '' stayed at number
one , as well as topping the Country Airplay and Country Streaming Songs charts . Together with the release of the deluxe edition of the album which also
reached number on Top Country Albums , and another number one for a new track `` Heaven '' on digital sales , Brown became the first artist to have simultaneous
number ones on all five main country charts . The song was certified Platinum by the RIAA on October 2 , 2017 . It was the third most - consumed country song of
2017 , with 1,551,000 including streams ( 599,000 in pure sale ) sold in the year . It has sold 679,000 copies in the US as of March 2018 . Music video ( edit )
The official music video for the song was directed by P.R. Brown and produced by Steve Lamar . The video was filmed along the Californian coast at The Inn at
Newport Ranch , a resort and cattle ranch to the north of San Francisco .
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
It features Brown and his duet partner Lauren Alaina singing along the coast . It was released on May 14 , 2017 . An earlier lyric video was first released on
May 4 , 2017 , and shows Brown and Alaina cruising down a road . Chart performance ( edit ) Weekly charts ( edit ) Chart ( 2017 ) Peak position Canada (
Canadian Hot 100 ) 45 Canada Country ( Billboard ) US Billboard Hot 100 26 US Country Airplay ( Billboard ) US Hot Country Songs ( Billboard ) Year - end charts
( edit ) Chart ( 2017 ) Position US Billboard Country Airplay 7 US Billboard Hot Country Songs US Billboard Hot 100 72 Certifications and sales ( edit ) Region
Certification Certified units / Sales Canada ( Music Canada ) Platinum 80,000 United States ( RIAA ) Platinum 679,000 sales figures based on certification alone
shipments figures based on certification alone References ( edit ) Jump up ^ Dukes , Billy ( January 24 , 2017 ) . `` Kane Brown ( Feat . Lauren Alaina ) , '
What Ifs ' ( Listen ) '' . Taste of Country .
</item 10>
=== END ITEM id=e65db654cfebf737 ===

=== ITEM id=9f7ef6a5b6a3ae4a ===
Gold sentences and facts:
Sentence 0: A defensive three - second violation , also known as illegal defense , is a basketball rules infraction in the National Basketball Association ( NBA ) introduced in the 2001 -- 2002 season . It is assessed when a member of the defending team spends more than three seconds in the free throw lane ( also known as the 16 - foot lane , or colloquially as `` in the paint '' ) while not actively guarding an opponent . To be considered actively guarding , a defender must be within arm 's length of an opponent and in a guarding position . A three - second count is suspended if :
  Fact 0.0: A defensive three-second violation is also known as illegal defense.
  Fact 0.1: A defensive three-second violation is a basketball rules infraction in the National Basketball Association (NBA).
  Fact 0.2: The defensive three-second rule was introduced in the 2001-2002 season.
  Fact 0.3: A defensive three-second violation is assessed when a member of the defending team spends more than three seconds in the free throw lane while not actively guarding an opponent.
  Fact 0.4: The free throw lane is also known as the 16-foot lane.
  Fact 0.5: The free throw lane is colloquially known as 'in the paint'.
  Fact 0.6: To be considered actively guarding, a defender must be within arm's length of an opponent.
  Fact 0.7: To be considered actively guarding, a defender must be in a guarding position.
  Fact 0.8: A three-second count is suspended under conditions listed after the word 'if'.

Retrieved items:
<item 1>
Defensive three-second violation
corners Motion Princeton Run and gun Grinnell System Nellie Ball Small ball Triangle Defense 1 - 2 - 1 - 1 1 -- 3 -- 1 2 -- 3 zone Amoeba Box - and - one Double
team Full - court press Hack - a-Shaq Jordan Rules Line Man - to - man Match - up zone Triangle - and - two Zone Equipment Backboard Ball Breakaway rim Court
Half court Key Net Possession arrow Whistle Clothing Basketball sleeve Finger sleeve Air Jordan Chuck Taylor All - Stars Miscellaneous 100 - point scorers 50 --
40 -- 90 club AAU All - Americans AAU champions Mikan Drill Player tracking Shootaround Tip drill Winning streaks This basketball - related article is a stub .
You can help Wikipedia by expanding it . Retrieved from `` https://en.wikipedia.org/w/index.php?title=Defensive_three-second_violation&oldid=803130800 ''
Categories : Basketball penalties Basketball stubs Hidden categories : Wikipedia articles needing clarification from November 2015 All stub articles Talk About
Wikipedia Français Edit links This page was last edited on 30 September 2017 , at 17 : 04 . About Wikipedia
</item 1>

<item 2>
2017–18 NBA season
The regular season started eight days earlier than the previous season . This move by the NBA aims for fewer `` back - to - back '' games , even fewer than the
previous season which had already reached an all - time low to help address teams that are `` resting '' their superstars . The NBA decreased the maximum number
of timeouts in a regulation game from 18 to 14 . The three team timeouts during the last two minutes of the game were also decreased to two per team in the last
three minutes to improve the flow and pace of the game . The trade deadline was moved from four days after the NBA All - Star Game to ten days before it , so
teams can settle their rosters before the All - Star break . Free - throw shooters will be assessed a delay - of - game violation if they venture outside of the
three - point line in between attempts . The NBA also added two - way contracted players , signifying an increasing partnership between the NBA and the newly
branded NBA G League . Referees can now assess dangerous closeouts of defenders to jump shooters with little space to land for a technical or a flagrant foul .
</item 2>

<item 3>
Defensive three-second violation
The WNBA started using it in 2013 . References ( edit ) Jump up ^ NBA Official Rules ( 2009 - 2010 ) Rule 10 , Section VII , B. Retrieved July 25 , 2010 Jump up
^ NBA Official Rules ( 2009 - 2010 ) Rule 10 , Section VII , D. Retrieved July 25 , 2010 Jump up ^ NBA Official Rules ( 2009 - 2010 ) Rule 10 , Section VII , E.
Retrieved July 25 , 2010 Jump up ^ NBA Official Rules ( 2009 - 2010 ) Rule 10 , Section VII , Penalty . Retrieved July 25 , 2010 Jump up ^ Blinebury , Fran (
June 8 , 2006 ) . `` Rule changes have NBA back in the fast lane '' . Houston Chronicle . Retrieved July 26 , 2010 . Jump up ^
http://www.sportingcharts.com/dictionary/nba/defensive-three-second-violation.aspx Jump up ^ Cruz , Agnes ( January 24 , 2003 ) . `` PBA get fresh look this
season '' . Arab News .
</item 3>

<item 4>
NBC Sunday Night Football
Then the banner returns to normal and show the change in the team 's score . Additionally ( beginning with Week 9 ) , timeout indicators were added below each
team 's respective scores . For the 2010 season , the timeout indicators were changed to three white trapezoids below the team abbreviations , and the play
clock was moved from above the team in possession of the football to above the game clock ( for the final two minutes of regulation and if necessary , overtime
) . The down markers also changed in 2010 , which is now featuring the team logo next to the down marker . 2012 -- 14 ( edit ) The scoring banner used from 2012
to 2014 , starting with the 2011 NFL Wild Card playoffs to Week 17 of the 2014 NFL regular season . Notice the addition of timeout indicators at the bottom ,
and the cleaner and larger looking font in the team 's initials .
</item 4>

<item 5>
American League
Most umpires made the switch to the inside protector , led by Don Denkinger in 1975 and Jim Evans the next year , although several did not , including Bill
Haller , Lou DiMuro , George Maloney , and Jerry Neudecker , who became the last MLB umpire to use the outside protector in 1985 . In 1994 , the league , along
with the National League , reorganized again , into three divisions ( East , West , and Central ) and added a third round to the playoffs in the form of the
American League Division Series , with the best second - place team advancing to the playoffs as a wild - card team , in addition to the three divisional
champions . In 1998 , the newly franchised Tampa Bay Devil Rays joined the league , and the Arizona Diamondbacks joined the National League : i.e. , each league
each added a fifteenth team . An odd number of teams per league meant that at least one team in each league would have to be idle on any given day , or
alternatively that odd team out would have had to play an interleague game against its counterpart in the other league .
</item 5>

<item 6>
2017–18 NBA season
Informally dubbed as the `` Zaza Pachulia rule '' . Also , officials can now assess if a player is in a shooting motion when determining if a perimeter foul is
a shooting or a common foul . The type of foul will be determined on the sequencing of the player 's movement if the foul happens before or after the player
starts a shooting motion . Additionally , they can also assess an offensive foul if an offensive player intentionally lunges into the defender or locks arms to
create contact , a rule informally dubbed as the `` Harden rule '' . On September 28 , 2017 , the NBA board of governors approved the changes to the draft
lottery system to discourage teams from losing on purpose . Starting with the 2018 -- 19 NBA season , the teams with the three worst records will have equal
odds at the # 1 pick . In addition , the revised draft lottery will consist of selecting the top four teams instead of the usual three that occurred for over
two decades . The approval for the change was nearly unanimous , with only the Oklahoma City Thunder voting against the change and the Dallas Mavericks
abstaining from voting .
</item 6>

<item 7>
Defensive three-second violation
`` PBA get fresh look this season '' . Arab News . Retrieved January 13 , 2014 . Jump up ^ `` WNBA Announces Rules Changes for 2013 Season '' . WNBA . Retrieved
Dec 16 , 2012 . Basketball ( outline ) General topics History of basketball James Naismith Variations of basketball Leagues Statistics Glossary of terms Index
of articles Rules Bonus Jump ball Officials Turnover Violation 3 seconds Offense Defense 5 seconds Basket interference Carrying Double dribble Goaltending Shot
clock Traveling Foul Flagrant Personal Technical Game play Air ball Alley - oop Assist Backboard shattering Ball hog Block Buzzer beater Cherry picking Dribble
Crossover Dunk Euro step Fadeaway Fast break Fly Flop Jump shot Layup Finger roll Field goal Four - point play Free throw Hook shot Moves Pick and roll
Positions Posterized Playbook Rebound Point Screen Back screen Slashing Steal Three - pointer Three - point play Uncontested shot Strategy General Sixth man
Offense Continuity offense Flex Shuffle UCLA High Post Wheel Dribble drive motion Four corners Motion Princeton Run and gun Grinnell System Nellie Ball Small
ball Triangle Defense 1
</item 7>

<item 8>
Defensive three-second violation
The offense receives one free throw and retains possession of the ball . Prior to the 2001 -- 2002 NBA season , any form of zone defense was considered an
illegal defense violation and resulted in a warning on the first violation and then a technical foul for any subsequent violations . However , the defensive
three - second violation makes it difficult for NBA defenses to play zone , since zone defenses usually position a player in the middle of the key to stop
penetration . The Philippine Basketball Association used to follow the illegal defense rule until the 2003 PBA season when it was abolished . The high school
game does not use this rule , nor does European basketball . The WNBA started using it in 2013 .
</item 8>

<item 9>
NBC Sunday Night Football
Additionally , the play clock appeared directly above the game clock throughout the entire game . Beginning with the 2014 Hall of Fame Game on August 3 , 2014 ,
the play clock was moved to the right side , next to the down / yardage graphic , of the bottom - screen score banner , which itself remained in the 4 : 3 safe
area . In addition , beginning with the NFL Kickoff Special on September 4 , 2014 , an electronic green - colored line - of - scrimmage marker was added to the
virtual on - field graphic . NBC 's Sunday Night Football was also the last of the five NFL broadcast partners to switch to a full 16 : 9 letterbox presentation
on its 4 : 3 standard - definition feed , a downscaled version of the HD feed 's native 16 : 9 format ( utilizing the Active Format Description # 10 flag ) ,
following Fox ( 2010 ) , ESPN 's Monday Night Football ( 2011 ) , NFL Network 's Thursday Night Football ( 2012 ) and CBS ( 2013 ) . Some of the graphics were
also re-positioned .
</item 9>

<item 10>
Defensive three-second violation
Defensive three - second violation - Wikipedia Defensive three - second violation Jump to : navigation , search A defensive three - second violation , also
known as illegal defense , is a basketball rules infraction in the National Basketball Association ( NBA ) introduced in the 2001 -- 2002 season . It is
assessed when a member of the defending team spends more than three seconds in the free throw lane ( also known as the 16 - foot lane , or colloquially as `` in
the paint '' ) while not actively guarding an opponent . To be considered actively guarding , a defender must be within arm 's length of an opponent and in a
guarding position . A three - second count is suspended if : a player is in the act of shooting there is a loss of team control the defender is actively
guarding an opponent the defender clears the painted area it is imminent the defender will become legal In addition , a player guarding an opponent with the
ball may be in the paint without actively guarding the opponent . The team committing a defensive three - second violation is assessed a team technical foul .
The offense receives one free throw and retains possession of the ball .
</item 10>
=== END ITEM id=9f7ef6a5b6a3ae4a ===

=== ITEM id=3d7ed0e0abb01194 ===
Gold sentences and facts:
Sentence 0: The Pakistani National Assembly ( Urdu : قومی اسمبلئ پاکستان ‬ ‎ -- Qaumī Asimbli'e Pākistān ) ; is the lower house of the bicameral Majlis - e-Shura , which also comprises the President of Pakistan and Senate ( upper house ) . The National Assembly and the Senate both convene at Parliament House in Islamabad . The National Assembly is a democratically elected body consisting of a total of 332 members who are referred to as Members of the National Assembly ( MNAs ) , of which 272 are directly elected members and 70 reserved seats for women and religious minorities . A political party must secure 172 seats to obtain and preserve a majority .
  Fact 0.0: The Pakistani National Assembly is the lower house of the bicameral Majlis-e-Shura.
  Fact 0.1: The Urdu name of the Pakistani National Assembly is Qaumī Asimbli'e Pākistān.
  Fact 0.2: The Majlis-e-Shura also comprises the President of Pakistan.
  Fact 0.3: The Majlis-e-Shura also comprises the Senate, which is the upper house.
  Fact 0.4: The National Assembly convenes at Parliament House in Islamabad.
  Fact 0.5: The Senate convenes at Parliament House in Islamabad.
  Fact 0.6: The National Assembly is a democratically elected body.
  Fact 0.7: The National Assembly has a total of 332 members.
  Fact 0.8: The members of the National Assembly are referred to as Members of the National Assembly (MNAs).
  Fact 0.9: 272 members of the National Assembly are directly elected.
  Fact 0.10: 70 members of the National Assembly hold reserved seats for women and religious minorities.
  Fact 0.11: A political party must secure 172 seats to obtain a majority.
  Fact 0.12: A political party must secure 172 seats to preserve a majority.

Retrieved items:
<item 1>
National Assembly of Pakistan
The Public Accounts Committee has a special role to review the report of the Auditor General . Senate , the upper house of the Parliament , has equal
representation from the federating units balancing the provincial inequality in the National Assembly , where the number of members is based on population of
the provinces . The Senate 's role is to promote national cohesion and harmony , and work as a stabilizing factor of the federation . The Senate numbers a total
of 104 members who serve six - year terms which are alternated so that half the senators are up for re-election by the electoral college every three years . The
National Assembly consists of 332 members . The Constitution empowers the President to dissolve the National Assembly , but the Senate is not subject to
dissolution . Only the Parliament can amend the Constitution , by two - thirds majority vote separately in each House . Members of National Assembly ( edit )
See also : List of Pakistan National Assembly Seats Qualifications ( edit ) The Constitution of Pakistan lists a number of requirements for members of the
National Assembly in Article 62 .
</item 1>

<item 2>
National Assembly of Pakistan
Seat distribution ( edit ) In the 14th Assembly , the seats have been allocated among the four provinces , federally - administered area , and capital territory
of Pakistan as follows : Province General Women Total Punjab 148 35 183 Sindh 61 14 75 KPK 35 8 43 Balochistan 14 17 FATA 12 - 12 Federal Capital - See also (
edit ) Speaker of National Assembly Senate of Pakistan Politics of Pakistan Prime Minister of Pakistan Constitution of Pakistan President of Pakistan Supreme
Court of Pakistan Pakistan Bar Council Punjab Bar Council Supreme Court Bar Association of Pakistan List of Pakistan National Assembly Seats List of provincial
governments of Pakistan showing provincial assemblies Notes ( edit ) ^ Jump up to : `` PML - N 's Ayaz Sadiq elected Speaker , Murtaza Abbasi Deputy Speaker in
NA '' . Dawn News . 3 June 2013 . Retrieved 6 June 2013 . Jump up ^ `` PPP 's Khursheed Shah declared leader of opposition in NA '' . Dawn News . 7 June 2013 .
Retrieved 7 June 2013 .
</item 2>

<item 3>
National Assembly of Pakistan
National Assembly of Pakistan - wikipedia National Assembly of Pakistan Jump to : navigation , search For List of MNAs elected in Pakistani general election ,
2013 , see List of MNAs elected in Pakistani general election , 2013 . National Assembly of Pakistan قومی اسمبلئ پاکستان ‬ 14th National Assembly of Pakistan
Type Type Lower house of the Parliament of Pakistan Leadership Speaker Ayaz Sadiq ( PML - N ) Since 9 November 2015 Deputy Speaker Javed Abbasi ( PML - N )
Since 3 June 2013 Leader of House Shahid Khaqan Abbasi ( PML - N ) Since 1 August 2017 Leader of Opposition Khurshid Shah ( PPP ) Since 7 June 2013 Structure
Seats 332 ( 3 vacant ) Political groups Government Coalition ( 209 ) PML -- N ( 188 ) JUI -- F ( 13 ) PML -- F ( 5 ) NPP ( 2 ) NP ( 1 ) Opposition ( 130 ) PPP (
47 ) PTI ( 32 ) MQM ( 24 ) JI ( 4 ) PkMAP (
</item 3>

<item 4>
National Assembly of Pakistan
The committees have also been empowered to invite or summon before it any member or any other person having a special interest in relation to any matter under
its consideration and may hear expert evidence and hold public hearing . Composition and elections ( edit ) The composition of the National Assembly is
specified in Article 51 of the Constitution of Pakistan . There are a total of 332 seats in the National Assembly . Of these , 272 are filled by direct
elections . In addition , the Pakistani Constitution reserves 10 seats for religious minorities and 60 seats for women , to be filled by proportional
representation among parties with more than 5 % of the vote . As of 2006 , there are 72 women members in the Assembly . Members of the National Assembly are
elected by the people in competitive multi-party elections , to be held at most five years apart on universal adult franchise . To be a member of electoral
college , according to Article 62 of the Constitution , candidates must be citizens of Pakistan and not less than 18 years of age . 2013 election ( edit ) Main
article : Pakistani general election , 2013 Results of the Pakistani general election ,
</item 4>

<item 5>
National Assembly of Pakistan
- 26 ( Bannu ) NA - 27 ( Lakki Marwat ) NA - 28 ( Buner ) NA - 29 ( Swat - I ) NA - 30 ( Swat - II ) NA - 31 ( Shangla ) NA - 32 ( Chitral ) NA - 33 ( Upper Dir
- cum - Lower Dir ) NA - 34 ( Lower Dir ) NA - 35 ( Malakand ) National Assembly constituencies from Islamabad and FATA ( NA - 36 to NA - 49 ) NA - 36 ( Tribal
Area - I ) NA - 37 ( Tribal Area - II ) NA - 38 ( Tribal Area - III ) NA - 39 ( Tribal Area - IV ) NA - 40 ( Tribal Area - V ) NA - 41 ( Tribal Area - VI ) NA -
42 ( Tribal Area - VII ) NA - 43 ( Tribal Area - VIII ) NA - 44 ( Tribal Area - IX ) NA - 45 ( Tribal Area - X ) NA - 46 ( Tribal Area - XI ) NA - 47 ( Tribal
Area - XII ) NA - 48 ( Islamabad - I ) NA - 49 ( Islamabad -
</item 5>

<item 6>
National Assembly of Pakistan
96 % 27 8 0 0 8 Total 45,388,404 100 % 266 266 60 10 336 Withheld : NA - 103 , NA - 237 ; Postponed : NA - 38 ; Re-poll ordered : NA - 46 ; Terminated : NA - 83
, NA - 254 Source : Detailed Results General Elections - 2013 , Reserve Seats Women , Reserve Seats Minorities Dissolution ( edit ) National Assembly can be
dissolved at the initiative of the Prime Minister . If dissolved , new elections are conducted for the Assembly . Article 58 of the Constitution of Pakistan
deals with the dissolution of the Assembly : 58 . Dissolution of the National Assembly : ( 1 ) The President shall dissolve the National Assembly if so advised
by the Prime Minister ; and the National Assembly shall , unless sooner dissolved , stand dissolved at the expiration of forty - eight hours after the Prime
Minister has so advised .
</item 6>

<item 7>
National Assembly of Pakistan
Area - XII ) NA - 48 ( Islamabad - I ) NA - 49 ( Islamabad - II ) National Assembly constituencies from the Punjab ( NA - 50 to NA - 197 ) NA - 50 ( Rawalpindi
- I ) NA - 51 ( Rawalpindi - II ) NA - 52 ( Rawalpindi - III ) NA - 53 ( Rawalpindi - IV ) NA - 54 ( Rawalpindi - V ) NA - 55 ( Rawalpindi - VI ) NA - 56 (
Rawalpindi - VII ) NA - 57 ( Attock - I ) NA - 58 ( Attock - II ) NA - 59 ( Attock - III ) NA - 60 ( Chakwal - I ) NA - 61 ( Chakwal - II ) NA - 62 ( Jhelum - I
) NA - 63 ( Jhelum - II ) NA - 64 ( Sargodha - I ) NA - 65 ( Sargodha - II ) NA - 66 ( Sargodha - III ) NA - 67 ( Sargodha - IV ) NA
</item 7>

<item 8>
National Assembly of Pakistan
631 0.47 % 0 9 National Peoples Party 197,829 0.44 % 0 10 Pakistan Muslim League ( Q ) 1,409,905 3.11 % 0 0 11 Awami National Party 453,057 1.00 % 0 0 12 Qaumi
Watan Party ( Sherpao ) 46,574 0.10 % 0 0 13 Pakistan Muslim League ( Z ) 128,510 0.28 % 0 0 14 National Party 61,148 0.13 % 0 0 15 Awami Muslim League Pakistan
93,046 0.20 % 0 0 16 Awami Jamhuri Ittehad Pakistan 71,175 0.16 % 0 0 17 Balochistan National Party 63,979 0.14 % 0 0 18 All Pakistan Muslim League 54,231 0.12
% 0 0 19 Independents 5,880,658 12.96 % 27 8 0 0 8 Total 45,388,
</item 8>

<item 9>
National Assembly of Pakistan
( 32 ) MQM ( 24 ) JI ( 4 ) PkMAP ( 3 ) ANP ( 2 ) PML -- Q ( 2 ) AJIP ( 1 ) AML ( 1 ) APML ( 1 ) BNP ( 1 ) PML -- Z ( 1 ) QWP ( 1 ) Ind ( 10 ) Elections Voting
system Mixed member majoritarian ( First past the post for most seats , 60 seats reserved for women and 10 seats reserved for religious minorities by
proportional representation ) Last election 11 May 2013 Meeting place Parliament House , Islamabad Website www.na.gov.pk The Pakistani National Assembly ( Urdu
: قومی اسمبلئ پاکستان ‬ ‎ -- Qaumī Asimbli'e Pākistān ) ; is the lower house of the bicameral Majlis - e-Shura , which also comprises the President of Pakistan
and Senate ( upper house ) . The National Assembly and the Senate both convene at Parliament House in Islamabad .
</item 9>

<item 10>
National Assembly of Pakistan
The National Assembly and the Senate both convene at Parliament House in Islamabad . The National Assembly is a democratically elected body consisting of a
total of 332 members who are referred to as Members of the National Assembly ( MNAs ) , of which 272 are directly elected members and 70 reserved seats for
women and religious minorities . A political party must secure 172 seats to obtain and preserve a majority . Members are elected through the first - past - the
- post system under universal adult suffrage , representing electoral districts known as National Assembly constituencies . According to the constitution , the
70 seats reserved for women and religious minorities are allocated to the political parties according to their proportional representation . Each National
Assembly is formed for a five - year term , commencing from the date of the first sitting , after which it is automatically dissolved . Currently the National
Assembly can not be dissolved by the President of Pakistan , it is dissolved by the Prime Minister of Pakistan . Election for 13th National Assembly was held on
18 February 2008 . On March 17 , 2013 13th National Assembly was dissolved on completion of its five - year term under Article 52 of the Constitution .
</item 10>
=== END ITEM id=3d7ed0e0abb01194 ===

=== ITEM id=e7e2d0c2d0a74ec0 ===
Gold sentences and facts:
Sentence 0: Much of the humus in most soils has persisted for more than 100 years , rather than having been decomposed into CO , and can be regarded as stable ; this organic matter has been protected from decomposition by microbial or enzyme action because it is hidden ( occluded ) inside small aggregates of soil particles , or tightly sorbed or complexed ) to clays . Most humus that is not protected in this way is decomposed within 10 years and can be regarded as less stable or more labile . Stable humus contributes few plant - available nutrients in soil , but it helps maintain its physical structure . A very stable form of humus is formed from the slow oxidation of soil carbon after the incorporation of finely powdered charcoal into the topsoil . This process is speculated to have been important in the formation of the very fertile Amazonian terra preta do Indio .
  Fact 0.0: Much of the humus in most soils has persisted for more than 100 years rather than having been decomposed into CO.
  Fact 0.1: The long-persisting humus in most soils can be regarded as stable.
  Fact 0.2: Much of the humus in most soils has been protected from decomposition by microbial or enzyme action.
  Fact 0.3: Protected humus is hidden (occluded) inside small aggregates of soil particles, or is tightly sorbed or complexed to clays.
  Fact 0.4: Most humus that is not protected in this way is decomposed within 10 years.
  Fact 0.5: Humus that is not protected in this way can be regarded as less stable or more labile.
  Fact 0.6: Stable humus contributes few plant-available nutrients in soil.
  Fact 0.7: Stable humus helps maintain the physical structure of soil.
  Fact 0.8: A very stable form of humus is formed from the slow oxidation of soil carbon after finely powdered charcoal is incorporated into the topsoil.
  Fact 0.9: The formation of the very stable humus from charcoal is speculated to have been important in the formation of the very fertile Amazonian terra preta do Indio.

Retrieved items:
<item 1>
Humus
Humus has many nutrients that improve the health of soil , nitrogen being the most important . The ratio of carbon to nitrogen ( C : N ) of humus is 10 : 1 .
Contents ( hide ) 1 Description 2 Humification 3 Stability 4 Horizons 5 Benefits of soil organic matter and humus 6 See also 7 References 8 External links
Description ( edit ) It is difficult to define humus precisely because it is a very complex substance which is not fully understood . Humus is different from
decomposing soil organic matter . The latter looks rough and has visible remains of the original plant or animal matter . Fully humified humus , on the contrary
, has a uniformly dark , spongy , and jelly - like appearance , and is amorphous ; it may gradually decompose over several years or persist for millennia . It
has no determinate shape , structure , or quality . However , when examined under a microscope , humus may reveal tiny plant , animal , or microbial remains
that have been mechanically , but not chemically , degraded . This suggests an ambiguous boundary between humus and soil organic matter .
</item 1>

<item 2>
Humus
Most soils have three major horizons : the surface horizon ( A ) , the subsoil ( B ) , and the substratum ( C ) . Some soils have an organic horizon ( O ) on
the surface , but this horizon can also be buried . The master horizon ( E ) is used for subsurface horizons that have significantly lost minerals ( eluviation
) . Bedrock , which is not soil , uses the letter R . Benefits of soil organic matter and humus ( edit ) The importance of chemically stable humus is thought by
some to be the fertility it provides to soils in both a physical and chemical sense , though some agricultural experts put a greater focus on other features of
it , such as its ability to suppress disease . It helps the soil retain moisture by increasing microporosity , and encourages the formation of good soil
structure . The incorporation of oxygen into large organic molecular assemblages generates many active , negatively charged sites that bind to positively
charged ions ( cations ) of plant nutrients , making them more available to the plant by way of ion exchange . Humus allows soil organisms to feed and reproduce
, and is often described as the `` life - force '' of the soil .
</item 2>

<item 3>
Humus
Soil Biology and Biochemistry . 26 ( 2 ) : 183 -- 220 . doi : 10.1016 / 0038 - 0717 ( 94 ) 90161 - 9 . Jump up ^ Humintech ® Definition Of Soil Organic Matter &
Humic Acids Based Products Jump up ^ Brady , N.C. ; Weil , R.R. ( 1999 ) . The nature and properties of soils . Upper Saddle River , N.J. : Prentice Hall , Inc
. Jump up ^ Soil biology Jump up ^ Berg , B. , McClaugherty , C. , 2007 . Plant litter : decomposition , humus formation , carbon sequestration , 2nd ed .
Springer , 338 pp. , ISBN 3 - 540 - 74922 - 5 Jump up ^ Levin , L. ; Forchiassin , F. ; Ramos , A.M. ( 2002 ) . `` Copper induction of lignin - modifying
enzymes in the white - rot fungus Trametes trogii '' . Mycologia. 94 : 377 -- 383 .
</item 3>

<item 4>
Humus
Stability ( edit ) Much of the humus in most soils has persisted for more than 100 years , rather than having been decomposed into CO , and can be regarded as
stable ; this organic matter has been protected from decomposition by microbial or enzyme action because it is hidden ( occluded ) inside small aggregates of
soil particles , or tightly sorbed or complexed ) to clays . Most humus that is not protected in this way is decomposed within 10 years and can be regarded as
less stable or more labile . Stable humus contributes few plant - available nutrients in soil , but it helps maintain its physical structure . A very stable
form of humus is formed from the slow oxidation of soil carbon after the incorporation of finely powdered charcoal into the topsoil . This process is speculated
to have been important in the formation of the very fertile Amazonian terra preta do Indio . Horizons ( edit ) Humus has a characteristic black or dark brown
color and is organic due to an accumulation of organic carbon . Soil scientists use the capital letters O , A , B , C , and E to identify the master horizons ,
and lowercase letters for distinctions of these horizons .
</item 4>

<item 5>
Humus
, 2011 . Accessed 24 Nov 2011 . < http://www.britannica.com/EBchecked/topic/276408/humus > . Jump up ^ Chertov , O.G. ; Kornarov , A.S. ; Crocker , G. ; Grace ,
P. ; Klir , J. ; Körschens , M. ; Poulton , P.R. ; Richter , D. ( 1997 ) . `` Simulating trends of soil organic carbon in seven long - term experiments using
the SOMM model of the humus types '' . Geoderma. 81 : 121 -- 135 . doi : 10.1016 / S0016 - 7061 ( 97 ) 00085 - 2 . Jump up ^ Baritz , R. , 2003 . Humus forms in
forests of the northern German lowlands . Schweizerbart , Stuttgart , Germany , 145 pp. ( 2 ) Jump up ^ Bunting , B.T. ; Lundberg , J. ( 1995 ) .
</item 5>

<item 6>
Humus
Humus is a colloidal substance and increases the cation exchange capacity of soil , hence its ability to store nutrients by chelation . While these nutrient
cations are available to plants , they are held in the soil and prevented from being leached by rain or irrigation . Humus can hold the equivalent of 80 -- 90 %
of its weight in moisture , and therefore increases the soil 's capacity to withstand drought . The biochemical structure of humus enables it to moderate , i.e.
buffer , excessive acidic or alkaline soil conditions . During humification , microbes secrete sticky , gum - like mucilages ; these contribute to the crumby
structure ( tilth ) of the soil by adhering particles together and allowing greater aeration of the soil . Toxic substances such as heavy metals and excess
nutrients can be chelated , i.e. , bound to the organic molecules of humus , and so prevented from leaching away . The dark , usually brown or black , color of
humus helps to warm cold soils in Spring .
</item 6>

<item 7>
Humus
Humus has a characteristic black or dark brown color and is an accumulation of organic carbon . Besides the three major soil horizons of ( A ) surface / topsoil
, ( B ) subsoil , and ( C ) substratum , some soils have an organic horizon ( O ) on the very surface . Hard bedrock ( R ) is not in a strict sense soil . In
soil science , humus ( derived in 1790 -- 1800 from the Latin humus for earth , ground ) denominates the fraction of soil organic matter that is amorphous and
without the `` cellular cake structure characteristic of plants , micro-organisms or animals . '' Humus significantly affects the bulk density of soil and
contributes to its retention of moisture and nutrients . In agriculture , `` humus '' sometimes also is used to describe mature or natural compost extracted
from a woodland or other spontaneous source for use as a soil conditioner . It is also used to describe a topsoil horizon that contains organic matter ( humus
type , humus form , humus profile ) . Humus is the dark organic matter that forms in soil when dead plant and animal matter decays .
</item 7>

<item 8>
Humus
The process that converts soil organic matter into humus feeds the population of microorganisms and other creatures in the soil , and thus maintains high and
healthy levels of soil life . The rate at which soil organic matter is converted into humus promotes ( when fast ) or limits ( when slow ) the coexistence of
plants , animals , and microorganisms in the soil . Effective humus and stable humus are additional sources of nutrients for microbes : the former provides a
readily available supply and the latter acts as a longeval storage reservoir . Decomposition of dead plant material causes complex organic compounds to be
slowly oxidized ( lignin - like humus ) or to decompose into simpler forms ( sugars and amino sugars , and aliphatic and phenolic organic acids ) , which are
further transformed into microbial biomass ( microbial humus ) or reorganized , and further oxidized , into humic assemblages ( fulvic and humic acids ) , which
bind to clay minerals and metal hydroxides . The ability of plants to absorb humic substances with their roots and metabolize them has been long debated . There
is now a consensus that humus functions hormonally rather than simply nutritionally in plant physiology .
</item 8>

<item 9>
Humus
This suggests an ambiguous boundary between humus and soil organic matter . While distinct , humus is an integral part of soil organic matter . Humification (
edit ) Microorganisms decompose a large portion of the soil organic matter into inorganic minerals that the roots of plants can absorb as nutrients . This
process is termed `` mineralization '' . In this process , nitrogen ( nitrogen cycle ) and the other nutrients ( nutrient cycle ) in the decomposed organic
matter are recycled . Depending on the conditions in which the decomposition occurs , a fraction of the organic matter does not mineralize , and instead is
transformed by a process called `` humification '' into concatenations of organic polymers . Because these organic polymers are resistant to the action of
microorganisms , they are stable , and constitute humus . This stability implies that humus integrates into the permanent structure of the soil , thereby
improving it . Humification can occur naturally in soil or artificially in the production of compost . Organic matter is humified by a combination of
saprotrophic fungi , bacteria , microbes and animals such as earthworms , nematodes , protozoa , and arthropods .
</item 9>

<item 10>
Humus
Plant remains , including those that animals digested and excreted , contain organic compounds : sugars , starches , proteins , carbohydrates , lignins , waxes
, resins , and organic acids . Decay in the soil begins with the decomposition of sugars and starches from carbohydrates , which decompose easily as
detritivores initially invade the dead plant organs , while the remaining cellulose and lignin decompose more slowly . Simple proteins , organic acids ,
starches , and sugars decompose rapidly , while crude proteins , fats , waxes , and resins remain relatively unchanged for longer periods of time . Lignin ,
which is quickly transformed by white - rot fungi , is one of the primary precursors of humus , together with by - products of microbial and animal activity .
The humus produced by humification is thus a mixture of compounds and complex biological chemicals of plant , animal , or microbial origin that has many
functions and benefits in soil . Some judge earthworm humus ( vermicompost ) to be the optimal organic manure .
</item 10>
=== END ITEM id=e7e2d0c2d0a74ec0 ===
