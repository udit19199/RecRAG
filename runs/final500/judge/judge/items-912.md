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

=== ITEM id=5a8a785004eef194 ===
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
Unsourced material may be challenged and removed . ( February 2011 ) ( Learn how and when to remove this template message ) ( Learn how and when to remove this
template message ) `` Like a Boss '' Song by The Lonely Island featuring Seth Rogen from the album Incredibad Recorded 2008 Genre Comedy hip hop Length 1 : 46
Label Universal Republic Records Songwriter ( s ) Andy Samberg , Akiva Schaffer , Jorma Taccone , A. Banks Producer ( s ) Rick tha Rular Incredibad track
listing `` Shrooms ( skit ) '' ( 9 ) `` Like a Boss '' ( 10 ) `` We Like Sportz '' ( 11 ) `` Like a Boss '' is a song written and recorded by comedy hip hop
troupe The Lonely Island for their debut studio album Incredibad . The song is a parody of the first official single from Slim Thug 's first album Already
Platinum , also named `` Like a Boss '' . The song had multiple versions , including an audio - only that did not have the celebrity interviewer and a clean
version that was produced in both audio and video .
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
In 2011 it was also subject to a lawsuit from the producers of the song who claimed they never received proper compensation for their work . Contents ( hide ) 1
Song 2 Video 3 Popularity 4 Lawsuit 5 Censorship 6 References 7 External links Song ( edit ) `` Like a Boss '' premiered as a Saturday Night Live Digital Short
on April 4 , 2009 . In the song , Andy Samberg is a businessman , evidently in middle management , having a performance review . The reviewer asks him to
describe the things he does in an average day , and does so for the remainder of the song in his trademark aggressive style while repeating the lyric `` like a
boss '' after every activity . He first lists activities which are fairly mundane , such as communicating to bigger bosses in corporate headquarters , approving
memorandums , remembering birthdays , leading a workshop , and promoting synergy . However , the song takes a foul turn after he mentions his female subordinate
Deborah , whom he repeatedly pursues , rejecting him , causing him to `` swallow sadness '' , face a sexual harassment lawsuit and be passed up for a promotion
.
</item 3>

<item 4>
Like a Boss
Like a Boss - wikipedia Like a Boss Jump to : navigation , search ( hide ) This article has multiple issues . Please help improve it or discuss these issues on
the talk page . ( Learn how and when to remove these template messages ) This article 's lead section does not adequately summarize key points of its contents .
Please consider expanding the lead to provide an accessible overview of all important aspects of the article . Please discuss this issue on the article 's talk
page . ( January 2015 ) This article needs additional citations for verification . Please help improve this article by adding citations to reliable sources .
Unsourced material may be challenged and removed .
</item 4>

<item 5>
Like a Boss
The video ends with the giant fish delivering the final words `` like a boss . '' Video ( edit ) The video features Seth Rogen , recording over the vocals done
by Akiva Schaffer on the single. The video adds additional visuals not included with the words of the song . It ends with Samberg 's tombstone being shown (
which has the epitaph `` He was da Best '' ) , and a skeptical Rogen trying to confirm some of the more bizarre events with Samberg . The `` bomb the Russians
'' part shows Samberg attacking St. Basil 's Cathedral . The video features brief cameo appearances from Saturday Night Live writer John Mulaney as the worker
that Samberg micromanages , Casey Wilson as the woman giving Samberg the memo , Kristen Wiig as Samberg 's office crush Debra , Jason Sudeikis as Samberg 's
boss and Bill Hader as the man on whom Samberg performs fellatio . It also features Jorma Taccone as the man giving him the lawsuit , and Akiva Schaffer selling
him the gun . Abby Elliott and Bobby Moynihan can be seen in the video as background dancers .
</item 5>

<item 6>
Like a Boss
Abby Elliott and Bobby Moynihan can be seen in the video as background dancers . Popularity ( edit ) Prior to the release of the video on Saturday Night Live ,
the term `` Like a Boss '' was not often searched for or used in social context . Though it was indeed searched for prior to the release of the video , the term
clearly spiked at release and peaked on Google in April 2009 , hitting a 100 on the Google Trend chart . It has also become popular on social media sites such
as Twitter Tumblr , and Facebook , where it is a popular hashtag to cite a task completed , goal achieved or successful action that someone is proud of doing .
It has also led to parody videos such as `` Like a Bus '' , where a bus is shown flying through traffic and often bad weather conditions with no regard or
slowing for the bad conditions . On 22 October 2012 , `` Like A Boss '' surpassed 100 million views . The video was featured by The Huffington Post , and TV
Guide . Lawsuit ( edit ) In November 2011 a production team from St. Louis filed charges against Andy Samberg and Co. , NBC Universal and Universal Republic
Records .
</item 6>

<item 7>
Like a Boss
Text is available under the Creative Commons Attribution - ShareAlike License ; additional terms may apply . By using this site , you agree to the Terms of Use
and Privacy Policy . Wikipedia ® is a registered trademark of the Wikimedia Foundation , Inc. , a non-profit organization . About Wikipedia
</item 7>

<item 8>
Like a Boss
`` Jack Sparrow '' `` 3 - Way ( The Golden Rule ) '' `` YOLO '' `` Spring Break Anthem '' `` Hugs '' Other songs `` Lazy Sunday '' `` Like a Boss '' `` Cool
Guys Do n't Look at Explosions '' `` Everything Is Awesome '' Film Hot Rod ( 2007 ) Popstar : Never Stop Never Stopping ( 2016 ) Television SNL Digital Shorts
Party Over Here Book Retrieved from `` https://en.wikipedia.org/w/index.php?title=Like_a_Boss&oldid=816720124 '' Categories : The Lonely Island songs 2009
singles Saturday Night Live sketches Comedy rap songs Hidden categories : Wikipedia introduction cleanup from January 2015 All pages needing cleanup Articles
covered by WikiProject Wikify from January 2015 All articles covered by WikiProject Wikify Articles needing additional references from February 2011 All
articles needing additional references Articles with multiple maintenance issues Articles with hAudio microformats Articles needing additional references from
January 2015 Talk Contents About Wikipedia Nederlands Simple English Edit links This page was last edited on 23 December 2017 , at 06 : 20 . Text is available
under the Creative Commons Attribution - ShareAlike License ; additional terms may apply .
</item 8>

<item 9>
Like a Boss
`` ' SNL ' Digital Short : ' Like A Boss ' ( With Video ) '' . The Ripcord . Archived from the original on 11 April 2009 . Retrieved 2014 - 06 - 21 . Jump up ^
`` Google Trends - websearch interest : like a boss - Worldwide , 2004 - present '' . Google Trends . Jump up ^ `` Twitter / search - # likeaboss '' .
Twitter.com . Jump up ^ `` like a boss on Tumblr '' . Tumblr.com . Jump up ^ `` # likeaboss '' . Facebook.com . Jump up ^ `` Like a BUS ! Playlist '' .
Youtube.com . Jump up ^ `` Like A Boss : SNL Digital Short ( VIDEO ) '' . The Huffington Post . 6 May 2009 . Retrieved 2014 - 06 - 23 . Jump up ^ DiNunno , Gina
. `` Today 's News : Our Take - Seth Rogen and Andy Samberg in SNL Digital Short `` Like a Boss '' `` . TV Guide.com . Retrieved 23 June 2014 .
</item 9>

<item 10>
Like a Boss
TV Guide.com . Retrieved 23 June 2014 . Jump up ^ Kaufman , Gill . `` ' SNL , ' Andy Samberg Sued By Producers over ' Like A Boss ' - MTV '' . MTV News .
Retrieved 11 July 2014 . External links ( edit ) Beard , Lanford ( November 8 , 2011 ) . `` NBC , Andy Samberg sued for copyright infringement '' .
Entertainment Weekly . Retrieved November 8 , 2011 . Original video : `` Like a Boss '' on YouTube Censored video : `` Like a Boss - Clean Version '' on YouTube
Video and Song Lyrics : `` Like a Boss '' Lonely Island 's Song Page : `` Like a Boss '' ( hide ) The Lonely Island Andy Samberg Akiva Schaffer Jorma Taccone
Discography Studio albums Incredibad Turtleneck & Chain The Wack Album Popstar : Never Stop Never Stopping Singles `` Dick in a Box '' `` Jizz in My Pants '' ``
I 'm on a Boat '' `` I Just Had Sex '' `` The Creep '' `` We 're Back ! ''
</item 10>
=== END ITEM id=5a8a785004eef194 ===

=== ITEM id=e39e74953d1161a0 ===
Gold sentences and facts:
Sentence 0: The total number of languages natively spoken in Africa is variously estimated ( depending on the delineation of language vs. dialect ) at between 1,250 to 2,100 , and by some counts at `` over 3,000 '' , Nigeria alone has over 500 languages ( according to the count of SIL Ethnologue ) , one of the greatest concentrations of linguistic diversity in the world .
  Fact 0.0: The total number of languages natively spoken in Africa is variously estimated at between 1,250 and 2,100, depending on the delineation of language vs. dialect.
  Fact 0.1: Some counts put the number of languages natively spoken in Africa at 'over 3,000'.
  Fact 0.2: Nigeria alone has over 500 languages, according to the count of SIL Ethnologue.
  Fact 0.3: Nigeria is one of the greatest concentrations of linguistic diversity in the world.

Retrieved items:
<item 1>
Languages of Africa
Languages of Africa - wikipedia Languages of Africa Jump to : navigation , search The traditional language families spoken in Africa : Afroasiatic Nilo -
Saharan Niger -- Congo Bantu Khoisan Indo - European Austronesian The languages of Africa are divided into six major language families : Afroasiatic languages
are spread throughout Western Asia , North Africa , the Horn of Africa and parts of the Sahel . Austronesian languages are spoken in Madagascar . Indo -
European languages are spoken in South Africa and Namibia ( Afrikaans , English , German ) and are used as lingua francas in the former colonies of Britain (
English ) , former colonies of France ( French ) , former colonies of Portugal ( Portuguese ) , former colonies of Spain ( Spanish ) and the current Spanish
territories of Ceuta , Melilla and the Canary Islands ( Spanish ) . Khoe languages are concentrated in the Kalahari Desert of Namibia and Botswana . Niger --
Congo languages ( Bantu and non-Bantu ) cover West , Central , Southeast and Southern Africa . Nilo - Saharan languages ( unity debated ) are spoken from
Tanzania to Sudan and from Chad to Mali .
</item 1>

<item 2>
Languages of Africa
English , French and Portuguese are important languages in Africa . About 130 million , 115 million and 30 million Africans , respectively , speak them as
either native or secondary languages . Portuguese has become the national language of Angola . Through ( among other factors ) sheer demographic weight ,
Africans are increasingly taking ownership of these three world languages and having an ever greater influence on their development and growth . Linguistic
features ( edit ) Some linguistic features are particularly common among languages spoken in Africa , whereas others are less common . Such shared traits
probably are not due to a common origin of all African languages . Instead , some may be due to language contact ( resulting in borrowing ) and specific idioms
and phrases may be due to a similar cultural background . Phonological ( edit ) Some widespread phonetic features include : certain types of consonants , such
as implosives ( / ɓa / ) , ejectives ( / kʼa / ) , the labiodental flap and in southern Africa , clicks ( / ǂa / , / ŋǃa / ) . True implosives are rare outside
Africa , and clicks and the flap almost unheard of .
</item 2>

<item 3>
Languages of Africa
135,000 native language of Mauritius Mossi Niger -- Congo 7,600,000 Recognised regional language in Burkina Faso Ndebele Niger -- Congo 1,090,000 Statuatory
national language in South Africa Northern Sotho Niger -- Congo 4,600,000 South Africa Oromo Afroasiatic 26,000,000 Ethiopia Portuguese Indo - European
13,700,000 ( estimated ) Angola , Cape Verde , Guinea - Bissau , Equatorial Guinea , Mozambique , São Tomé and Príncipe Sesotho Niger -- Congo 5,600,000 Lesotho
, South Africa , Zimbabwe Shona Niger -- Congo 14,200,000 incl . Manyika , Ndau ( 2000 -- 2006 ) Zimbabwe Somali Afroasiatic 16,600,000 Somalia Spanish Indo -
European 2,700,000 Equatorial Guinea Swahili Niger -- Congo 15,000,000 official in Tanzania , Kenya , Uganda , Rwanda national language of Democratic Republic
of Congo Tigrinya Afroasiatic 7,000,000 Eritrea Tshiluba Niger -- Congo 6,300,
</item 3>

<item 4>
Languages of Africa
Case , tense and other categories may be distinguished only by tone . Semantic ( edit ) Quite often , only one term is used for both animal and meat ; the word
nama or nyama for animal / meat is particularly widespread in otherwise widely divergent African languages . Number of speakers ( edit ) The following is a
table displaying the number of speakers of given languages within Africa : Language Family Native speakers ( L1 ) Official status per country Afrikaans Indo -
European 7,200,000 South Africa Akan Niger -- Congo 11,000,000 None . Government sponsored language of Ghana Amharic Afroasiatic 21,800,000 Ethiopia Arabic
Afroasiatic 150,000,000 but with separate mutually unintelligible varieties Algeria , Chad , Comoros , Djibouti , Egypt , Eritrea , Libya , Mauritania , Morocco
, Somalia , Sudan , Tunisia Berber Afroasiatic 56,000,000 ( estimated ) ( including separate unintelligible varieties ) Morocco , Algeria Chewa Niger -- Congo
9,700,000 Malawi , Zimbabwe English Indo - European 6,500,
</item 4>

<item 5>
Languages of Africa
5 Unclassified languages 1.4. 6 Sign languages 2 Language in Africa 2.1 Official Languages 2.2 Cross-border languages 2.3 Language change and planning 2.4
Demographics 3 Linguistic features 3.1 Phonological 3.2 Syntactic 3.3 Semantic 4 Number of speakers 5 See also 5.1 General 5.2 Works 5.3 Classifiers 6 Notes 7
References 8 External links Language groups ( edit ) Clickable map showing the traditional language families , subfamilies and major languages spoken in Africa
Most languages spoken in Africa belong to one of three large language families : Afroasiatic , Nilo - Saharan and Niger -- Congo . Another hundred belong to
small families such as Ubangian ( sometimes grouped within Niger - Congo ) and the various families called Khoisan , or the Indo - European and Austronesian
language families mainly spoken outside Africa ; the presence of the latter two dates to 2,600 and 1,500 years ago , respectively . In addition , the languages
of Africa languages include several unclassified languages and sign languages .
</item 5>

<item 6>
Languages of Africa
Trade languages are another age - old phenomenon in the African linguistic landscape . Cultural and linguistic innovations spread along trade routes and
languages of peoples dominant in trade developed into languages of wider communication ( lingua franca ) . Of particular importance in this respect are Berber (
North and West Africa ) , Jula ( western West Africa ) , Fulfulde ( West Africa ) , Hausa ( West Africa ) , Lingala ( Congo ) , Swahili ( Southeast Africa ) ,
Somali ( Horn of Africa ) and Arabic ( North Africa and Horn of Africa ) . After gaining independence , many African countries , in the search for national
unity , selected one language , generally the former colonial language , to be used in government and education . However , in recent years , African countries
have become increasingly supportive of maintaining linguistic diversity . Language policies that are being developed nowadays are mostly aimed at
multilingualism . Official languages ( edit ) Official languages in Africa : Afrikaans Portuguese Arabic Spanish English Swahili French other languages Besides
the former colonial languages of English , French , Portuguese and Spanish , the following languages are official at the national level in Africa (
non-exhaustive list ) : Afroasiatic Arabic in Comoros , Chad ,
</item 6>

<item 7>
Languages of Africa
In addition , the languages of Africa languages include several unclassified languages and sign languages . The earliest Afroasiatic languages are associated
with the Capsian culture , the Nilo - Saharan languages are linked with the Khartoum Mesolithic / Neolithic , the Niger - Congo languages are correlated with
the west and central African hoe - based farming traditions and the Khoisan languages are matched with the south and southeastern Wilton industries . More
broadly , the Afroasiatic family is tentatively grouped within the Nostratic superfamily , and the Nilo - Saharan and Niger - Congo phyla form the Niger -
Saharan macrophylum . Afroasiatic languages ( edit ) Main article : Afroasiatic languages Afroasiatic languages are spoken throughout North Africa , the Horn of
Africa , West Asia and parts of the Sahel . There are approximately 375 Afroasiatic languages spoken by over 350 million people . The main subfamilies of
Afroasiatic are the Berber languages , Semitic languages , Chadic languages and the Cushitic languages . The Afroasiatic Urheimat is uncertain .
</item 7>

<item 8>
Languages of Africa
There are several other small families and language isolates , as well as languages that have yet to be classified . In addition , Africa has a wide variety of
sign languages , many of which are language isolates ( see below ) . The total number of languages natively spoken in Africa is variously estimated ( depending
on the delineation of language vs. dialect ) at between 1,250 to 2,100 , and by some counts at `` over 3,000 '' , Nigeria alone has over 500 languages (
according to the count of SIL Ethnologue ) , one of the greatest concentrations of linguistic diversity in the world . Around a hundred languages are widely
used for inter-ethnic communication . Arabic , Somali , Berber , Amharic , Oromo , Igbo , Swahili , Hausa , Manding , Fulani and Yoruba are spoken by tens of
millions of people . Twelve dialect clusters ( which may group up to a hundred linguistic varieties ) are spoken by 75 percent , and fifteen by 85 percent , of
Africans as a first or additional language .
</item 8>

<item 9>
Languages of Africa
000,000 Eritrea Tshiluba Niger -- Congo 6,300,000 ( 1991 ) national language of Democratic Republic of the Congo Tswana Niger -- Congo 5,800,000 South Africa ,
Botswana Umbundu Niger -- Congo 6,000,000 recognised national language of Angola Xhosa Niger -- Congo 7,600,000 South Africa , Zimbabwe Yoruba Niger -- Congo
28,000,000 Zulu Niger -- Congo 10,400,000 South Africa See also ( edit ) Africa portal Language portal General ( edit ) Languages of the African Union Writing
systems of Africa Journal of West African Languages Works ( edit ) Polyglotta Africana The Languages of Africa Classifiers ( edit ) Karl Lepsius Wilhelm Bleek
Carl Meinhof Diedrich Westermann Joseph Greenberg Notes ( edit ) Jump up ^ Heine , Bernd ; Heine , Bernd , eds. ( 2000 ) . African Languages : an Introduction .
Cambridge University Press . Jump up ^ Epstein , Edmund L. ; Kole , Robert , eds. ( 1998 ) . The Language of African Literature . Africa World Press .
</item 9>

<item 10>
Languages of Africa
In addition to the ( likely modest ) impact of borders , there are also cases of dialect levelling ( such as in Igbo and probably many others ) , koinés ( such
as N'Ko and possibly Runyakitara ) and emergence of new dialects ( such as Sheng ) . In some countries , there are official efforts to develop standardized
language versions . There are also many less widely spoken languages that may be considered endangered languages . Demographics ( edit ) Further information :
Demographics of Africa Of the 1 billion Africans ( in 2009 ) , about 17 percent speak an Arabic dialect . About 10 percent speak Swahili , the lingua franca of
Southeast Africa ; about 5 percent speak a Berber dialect ; and about 5 percent speak Hausa , which serves as a lingua franca in much of the Sahel . Other
important West African languages are Yoruba , Igbo and Fula . Major Horn of Africa languages are Amharic , Oromo and Somali . Important South African languages
are Zulu , Xhosa and Afrikaans . English , French and Portuguese are important languages in Africa .
</item 10>
=== END ITEM id=e39e74953d1161a0 ===

=== ITEM id=d2b5c93a9ad63c71 ===
Gold sentences and facts:
Sentence 0: On the November 19 , 1999 , episode of Millionaire , Carpenter proceeded to advance to the million - dollar question without using any lifelines . He then used his Phone - A-Friend to call his father not for help , but rather to tell him he was going to win the game . Carpenter answered the question correctly and became the show 's first millionaire . His win gave him national recognition and led to multiple talk show appearances , as well as reappearances on Millionaire itself .
  Fact 0.0: On the November 19, 1999 episode of Millionaire, Carpenter advanced to the million-dollar question.
  Fact 0.1: Carpenter advanced to the million-dollar question without using any lifelines.
  Fact 0.2: Carpenter used Phone-A-Friend to call his father.
  Fact 0.3: Carpenter did not call his father for help with the question.
  Fact 0.4: Carpenter called his father to tell him he was going to win the game.
  Fact 0.5: Carpenter answered the million-dollar question correctly.
  Fact 0.6: Carpenter became the show's first millionaire.
  Fact 0.7: Carpenter's win gave him national recognition.
  Fact 0.8: Carpenter's win led to multiple talk show appearances.
  Fact 0.9: Carpenter's win led to reappearances on Millionaire itself.

Retrieved items:
<item 1>
John Carpenter (game show contestant)
He later participated as part of the Mob on NBC 's 1 vs. 100 on October 27 , 2006 , and as a contestant on the Game Show Network game show Grand Slam .
Carpenter appeared on the August 16 , 2009 episode of Millionaire in prime time for its tenth anniversary . In the audience with him were his father , his wife
, and his son . Additionally , he was also the first expert in the `` Ask the Expert '' lifeline for the eighth season of the syndicated series . See also (
edit ) Biography portal List of Who Wants to Be a Millionaire ? top prize winners References ( edit ) Jump up ^ `` Millionaire Makeovers ! '' . People . Vol. 54
no . 20 . Time Inc . November 13 , 2000 . Archived from the original on September 11 , 2015 . Retrieved July 8 , 2015 . Jump up ^ Petrozello , Donna ( February
3 , 2000 ) . `` Million - plu $ reply an educated guess '' . New York Daily News . p. 101 .
</item 1>

<item 2>
John Carpenter (game show contestant)
Retrieved September 6 , 2009 . ^ Jump up to : `` Who Wants to Be a Millionaire '' . Gameshow Hall of Fame . GSN . January 21 , 2007 . ^ Jump up to : Vigoda ,
Arlene ( November 22 , 1999 ) . `` Million - dollar winner untaxed by celebrity '' . USA Today . Gannett Company . p. 1D . Jump up ^ `` Taxman scoops a million
'' . BBC News . November 21 , 1999 . Archived from the original on June 21 , 2015 . Retrieved August 1 , 2014 . Jump up ^ Davis , Kristin ( November 2000 ) . ``
I 'm Rich ! ( Now what ? ) '' . Kiplinger 's Personal Finance . pp. 93 -- 94 . ^ Jump up to : Duca , Lauren ( August 15 , 2014 ) . `` The Final Answer On Who
Wants To Be A Millionaire ? , 15 Years After It Premiered '' .
</item 2>

<item 3>
John Carpenter (game show contestant)
`` Miami ' Millionaire ' gets 2nd shot at final answer '' . St. Petersburg Times . Archived from the original on April 5 , 2008 . Retrieved October 9 , 2014 .
Jump up ^ Bracht , Mel ( May 26 , 2000 ) . `` Money is n't the root of happiness : Tulsan has more fun in 2nd Millionaire gig '' . NewsOK . Oklahoma Publishing
Company . Archived from the original on March 13 , 2017 . Retrieved October 9 , 2014 . Jump up ^ Super Millionaire . Season 1 . Episode 2 . February 23 , 2004 .
ABC . Jump up ^ 1 vs. 100 . Season 1 . Episode 3 . October 27 , 2006 . NBC . Jump up ^ Owen , Rob ( August 2 , 2007 ) . `` Dennis Miller rallies around game
show all - stars '' . Pittsburgh Post-Gazette . p . WE - 33 .
</item 3>

<item 4>
John Carpenter (game show contestant)
He explained to Kiplinger 's Personal Finance that `` after the taxes , it 's not change - your - life kind of money if you want to eat every day . '' Carpenter
also described the fame as having a bigger impact on his life than the money , later stating : `` The money does n't change your life . What happens afterwards
might . '' Later work ( edit ) Shortly after winning on Millionaire , Carpenter played himself in a Saturday Night Live skit . Donald Trump , played by Darrell
Hammond , announced that Carpenter would be his running - mate in the presidential election . Afterward , Carpenter pretended to call his father , then shouted
, `` Live from New York , it 's Saturday Night ! '' Carpenter also appeared on Good Morning America , Oprah Winfrey Show , Live with Regis and Kathie Lee , and
Late Show with David Letterman . Carpenter appeared as himself in the second half of the fourth season of Oz . He plays a contestant in a fictional TV game show
called Up Your Ante that the prisoners in Em City are watching .
</item 4>

<item 5>
John Carpenter (game show contestant)
The $1 million question was , `` Which of these U.S. Presidents appeared on the television series ' Laugh - In ' ? '' , with the choices being A ) Lyndon
Johnson , B) Richard Nixon , C ) Jimmy Carter , and D ) Gerald Ford . Carpenter used his Phone - A-Friend lifeline to call his father not for help , but rather
to inform him that he knew the answer . Carpenter later explained , `` I thought I 'd look so cocky if I did n't use any lifelines , so I faked it . '' With his
win , Carpenter became the first contestant in the worldwide Millionaire franchise to win the show 's top prize . He said that the only question that had
flustered him was one which asked for the location of the Gunfight at the O.K. Corral . Carpenter eventually remembered that the film Tombstone included the
gunfight , and he replied correctly with the answer ' Tombstone , Arizona ' . While taking a vacation after his win , Carpenter considered quitting his job with
the IRS , but eventually decided against it .
</item 5>

<item 6>
John Carpenter (game show contestant)
The show within the show is hosted by Gordon Elliott , with Eartha Kitt and Didi Conn appearing as celebrity participants . With Rod L. Evans , Carpenter
co-authored a trivia book titled Matching Wits With the Million - Dollar Mind : The World 's Hardest Trivia Quizzes From America 's First Quiz Show Millionaire
. The book was published by Berkley Books in 2002 . Other game show appearances ( edit ) In 2000 , Carpenter appeared in the Who Wants to Be a Millionaire
Champions Edition , in which previous contestants who won $250,000 -- $1,000,000 played again , with half of their additional winnings from the Champions
Edition going to a charity of their choice . Carpenter won $250,000 , bringing his total Millionaire winnings to $1,250,000 . In 2004 , Carpenter participated
in Super Millionaire , as one of the `` Three Wise Men '' on the episode during which Robert `` Bob - O '' Essig won $1,000,000 . He later participated as part
of the Mob on NBC 's 1 vs.
</item 6>

<item 7>
John Carpenter (game show contestant)
John Carpenter ( game show contestant ) - wikipedia John Carpenter ( game show contestant ) Jump to : navigation , search John Carpenter 1968 ( age 48 -- 49 )
Education Rutgers University Occupation IRS agent Known for First top prize winner on Who Wants to Be a Millionaire Home town Northampton , Massachusetts John
Carpenter ( born c. 1968 ) is an American game show contestant and IRS agent . He is best known for becoming the first top prize winner on the United States
version of Who Wants to Be a Millionaire . He held the record for the largest single win in United States game show history , until it was broken by Rahim
Oberholtzer who won $1.12 million on another U.S. quiz show , Twenty One . Carpenter was also the first top prize winner among all international versions of the
Who Wants to Be a Millionaire ? series . On the November 19 , 1999 , episode of Millionaire , Carpenter proceeded to advance to the million - dollar question
without using any lifelines . He then used his Phone - A-Friend to call his father not for help , but rather to tell him he was going to win the game .
</item 7>

<item 8>
John Carpenter (game show contestant)
Carpenter answered the question correctly and became the show 's first millionaire . His win gave him national recognition and led to multiple talk show
appearances , as well as reappearances on Millionaire itself . Contents ( hide ) 1 Career and family 2 Who Wants to Be a Millionaire 3 Later work 3.1 Other game
show appearances 4 See also 5 References 6 External links Career and family ( edit ) Carpenter is from Northampton , Massachusetts . His father , Tom , worked
as a computer program analyst for the Department of Veterans Affairs , while his mother , Gail , served as an administrative assistant for the Massachusetts
Audubon Society . In 1986 , he enrolled at Rutgers University and graduated in 1990 with a degree in economics . In January 1991 , he joined the Internal
Revenue Service after completing government exams and tests . In November 1996 , Carpenter met his future wife , Deborah , who was studying for a master 's
degree at Southern Connecticut State University and worked as a manager of a Fleet Bank branch in New Haven . They married in August 1998 .
</item 8>

<item 9>
John Carpenter (game show contestant)
, 15 Years After It Premiered '' . Huffington Post . Archived from the original on August 22 , 2015 . Retrieved July 9 , 2015 . Jump up ^ de Moraes , Lisa (
November 23 , 1999 ) . `` Drab ' Millionaire ' Winner Emerges as a Media Animal '' . The Washington Post . p . C07 . Jump up ^ Moore , Frazier ( January 6 ,
2001 ) . `` Wizard of ' Oz ' lives with permanent reminder of gritty show '' . The San Diego Union Tribune . p . E6 . Jump up ^ `` Matching Wits with the
Million - Dollar Mind : The World ; s Hardest Trivia Quizzes from America 's First Quiz Show Millionaire '' . The Book Depository . Archived from the original
on October 25 , 2015 . Retrieved October 24 , 2015 . Jump up ^ Deggans , Eric ( May 12 , 2000 ) . `` Miami ' Millionaire ' gets 2nd shot at final answer '' .
St.
</item 9>

<item 10>
John Carpenter (game show contestant)
They married in August 1998 . At the time of his appearance on Millionaire , he was 31 years old . When he revealed his profession as an IRS officer on
Millionaire , Carpenter was playfully booed by the audience . Who Wants to Be a Millionaire ( edit ) $1 million ( 15 of 15 ) - no time limit Which of these U.S.
Presidents appeared on the television series `` Laugh - In '' ? A : Lyndon Johnson B : Richard Nixon C : Jimmy Carter D: Gerald Ford Carpenter was originally
uninterested in Millionaire but eventually tuned in one night after dinner while having friends over at his house . When he found the show 's higher - tier
questions to not be , in his words , `` any more difficult '' than the lower - tier ones , he decided to call in to the show 's hotline for a chance to become a
contestant . Carpenter answered all of the hotline questions correctly and was on the show within two days . Host Regis Philbin described Carpenter as having ``
cruised right through those first fourteen questions , '' he had proceeded to reach the final question without using any of his lifelines .
</item 10>
=== END ITEM id=d2b5c93a9ad63c71 ===

=== ITEM id=972efe7fc241bbac ===
Gold sentences and facts:
Sentence 0: The College Football Playoff selection committee chose the semifinalists following the conclusion of the 2017 regular season . Alabama and Georgia advanced to the national championship after winning the semifinal games hosted by the Sugar Bowl and the Rose Bowl , respectively , on January 1 , 2018 . The championship game was played at Mercedes - Benz Stadium in Atlanta , Georgia on January 8 , 2018 .
  Fact 0.0: The College Football Playoff selection committee chose the semifinalists.
  Fact 0.1: The semifinalists were chosen following the conclusion of the 2017 regular season.
  Fact 0.2: Alabama advanced to the national championship.
  Fact 0.3: Georgia advanced to the national championship.
  Fact 0.4: Alabama won its semifinal game hosted by the Sugar Bowl on January 1, 2018.
  Fact 0.5: Georgia won its semifinal game hosted by the Rose Bowl on January 1, 2018.
  Fact 0.6: The championship game was played at Mercedes-Benz Stadium in Atlanta, Georgia.
  Fact 0.7: The championship game was played on January 8, 2018.
Sentence 1: The Mercedes - Benz Stadium in Atlanta , Georgia was announced as the host site for the fourth College Football National Championship on November 4 , 2015 .
  Fact 1.0: Mercedes-Benz Stadium in Atlanta, Georgia was announced as the host site for the fourth College Football National Championship.
  Fact 1.1: The host site announcement was made on November 4, 2015.

Retrieved items:
<item 1>
2018 College Football Playoff National Championship
1982 1984 1985 1986 1988 1989 1990 1991 1992 1993 1994 1995 1997 1998 1999 2000 2001 2002 2003 2005 2006 2007 2008 2009 2011 2012 2013 2014 2015 2016 2017 2018
Division III 1973 1976 1977 1978 1979 1980 1981 1982 1984 1985 1986 1988 1989 1990 1991 1992 1993 1994 1995 1997 1998 1999 2000 2001 2002 2003 2005 2006 2007
2008 2009 2011 2012 2013 2014 2015 2016 2017 2018 Games played in January are denoted by an asterisk Alabama Crimson Tide bowl games 1926 Rose Bowl 1927 Rose
Bowl 1931 Rose Bowl
</item 1>

<item 2>
2018 College Football Playoff National Championship
Alabama and Georgia advanced to the national championship after winning the semifinal games hosted by the Sugar Bowl and the Rose Bowl , respectively , on
January 1 , 2018 . The championship game was played at Mercedes - Benz Stadium in Atlanta , Georgia on January 8 , 2018 . Contents ( hide ) 1 Background 2 2018
College Football Playoff 3 Teams 3.1 Alabama 3.2 Georgia 4 Game summary 4.1 Scoring summary 4.2 Statistics 5 Broadcasting 6 Dispute 7 See also 8 References 9
External links Background ( edit ) The Mercedes - Benz Stadium in Atlanta , Georgia was announced as the host site for the fourth College Football National
Championship on November 4 , 2015 .
</item 2>

<item 3>
2018 College Football Playoff National Championship
2018 College football Playoff ( edit ) Semifinals Championship January 1 -- Sugar Bowl Mercedes - Benz Superdome , New Orleans 1 Clemson 6 4 Alabama 24 January
8 -- National Championship Mercedes - Benz Stadium , Atlanta 4 Alabama ( OT ) 26 January 1 -- Rose Bowl Rose Bowl , Pasadena 3 Georgia 23 2 Oklahoma 48 3
Georgia ( 2OT ) 54 This bracket : view talk edit The College Football Playoff ( CFP ) selected four teams to advance to the playoff : the Clemson Tigers ,
Oklahoma Sooners , Georgia Bulldogs , and Alabama Crimson Tide . Teams ( edit ) Alabama ( edit ) See also : 2017 Alabama Crimson Tide football team The Alabama
Crimson Tide defeated the Clemson Tigers in the 2018 Sugar Bowl by a score of 24 -- 6 to reach the championship game . The Tide had a 12 -- 1 season , becoming
the SEC West Division co-champions with the Auburn Tigers . The Crimson Tide were coached by Nick Saban .
</item 3>

<item 4>
2018 College Football Playoff National Championship
champions Mythical national championship Plus - One system Power Five conferences Championship games for each season are played in January , while semifinal
games are identified by season year , alternating between December and January Links to related articles 2017 -- 18 NCAA football bowl game season New Mexico (
Dec. 16 ) Las Vegas ( Dec. 16 ) Camellia ( Dec. 16 ) Cure ( Dec. 16 ) New Orleans ( Dec. 16 ) Boca Raton ( Dec. 19 ) Frisco ( Dec. 20 ) Gasparilla ( Dec. 21 )
Famous Idaho Potato ( Dec. 22 ) Bahamas ( Dec. 22 ) Armed Forces ( Dec. 23 ) Birmingham ( Dec. 23 ) Dollar General ( Dec. 23 ) Hawaii ( Dec. 24 ) Quick Lane (
Dec. 26 ) Heart of Dallas ( Dec. 26 ) Cactus ( Dec. 26 ) Independence ( Dec. 27 ) Pinstripe ( Dec. 27 ) Texas ( Dec. 27 ) Foster Farms ( Dec. 27 ) Military (
Dec. 28 ) Camping World ( Dec. 28 ) Alamo ( Dec.
</item 4>

<item 5>
2018 College Football Playoff National Championship
related to 2018 College Football Playoff National Championship at Wikimedia Commons College Football Playoff Championships Overview College Football Playoff
National Championship Games 2014 -- 15 2015 -- 16 2016 -- 17 2017 -- 18 2018 -- 19 2019 -- 20 2020 -- 21 Semifinal Bowls Rose 2014 ( Jan 2015 ) 2017 ( Jan 2018
) 2020 ( Jan 2021 ) Sugar 2014 ( Jan 2015 ) 2017 ( Jan 2018 ) 2020 ( Jan 2021 ) Orange 2015 ( Dec 2015 ) 2018 ( Dec 2018 ) Cotton 2015 ( Dec 2015 ) 2018 ( Dec
2018 ) Peach 2016 ( Dec 2016 ) 2019 ( Dec 2019 ) Fiesta 2016 ( Dec 2016 ) 2019 ( Dec 2019 ) Other List of College Football Playoff games New Year 's Six College
Football Playoff National Championship Trophy NCAA Division I FBS champions Mythical national championship Plus - One system Power Five conferences Championship
games for each season are played in
</item 5>

<item 6>
2018 College Football Playoff National Championship
Jump up ^ Alex Scarbrough ( November 4 , 2015 ) . `` Atlanta , Santa Clara and New Orleans land CFP title games for 2018 -- 20 '' . ESPN.com . Retrieved
November 4 , 2015 . ^ Jump up to : `` Selection Committee Rankings : Final Top 25 Rankings '' ( PDF ) . College Football Playoff . December 3 , 2017 . Retrieved
December 3 , 2017 . Jump up ^ `` Alabama 's Sugar Bowl win against Clemson sets up all - SEC final : Final score , stats , analysis '' . nola.com . January 1 ,
2018 . Retrieved January 2 , 2018 . Jump up ^ Emerson , Seth ( January 1 , 2018 ) . `` Georgia to play for national championship after Rose Bowl win in 2 OT ''
. ajc.com . Retrieved January 2 , 2018 . ^ Jump up to : Hoffman , Benjamin ; Drape , Joe ; Tracy , Marc ( January 8 , 2018 ) . `` National Championship Game :
Georgia vs.
</item 6>

<item 7>
2018 College Football Playoff National Championship
Not an officially sanctioned NCAA championship Retrieved from ``
https://en.wikipedia.org/w/index.php?title=2018_College_Football_Playoff_National_Championship&oldid=822294017 '' Categories : 2017 -- 18 NCAA football bowl
games 2017 -- 18 College Football Playoff 2018 in sports in Georgia ( U.S. state ) Sports competitions in Atlanta American football in Georgia ( U.S. state )
2010s in Atlanta Alabama Crimson Tide football bowl games Georgia Bulldogs football bowl games January 2018 sports events in the United States Hidden categories
: Use mdy dates from January 2018 Talk Contents About Wikipedia Français Edit links This page was last edited on 25 January 2018 , at 15 : 15 . Text is
available under the Creative Commons Attribution - ShareAlike License ; additional terms may apply . By using this site , you agree to the Terms of Use and
Privacy Policy . Wikipedia ® is a registered trademark of the Wikimedia Foundation , Inc. , a non-profit organization . About Wikipedia
</item 7>

<item 8>
2018 College Football Playoff National Championship
28 ) Camping World ( Dec. 28 ) Alamo ( Dec. 28 ) Holiday ( Dec. 28 ) Belk ( Dec. 29 ) Cotton ( Dec. 29 ) Sun ( Dec. 29 ) Music City ( Dec. 29 ) Arizona ( Dec.
29 ) Fiesta ( Dec. 30 ) Liberty ( Dec. 30 ) Orange ( Dec. 30 ) TaxSlayer ( Dec. 30 ) Citrus ( Jan. 1 ) Outback ( Jan. 1 ) Peach ( Jan. 1 ) College Football
Playoff games : Rose ( semifinal ) ( Jan. 1 ) Sugar ( semifinal ) ( Jan. 1 ) National Championship Game ( Jan. 8 ) All - Star Games East -- West Shrine Game (
Jan. 20 ) NFLPA Collegiate Bowl ( Jan. 20 ) Senior Bowl ( Jan. 27 ) NCAA Football Championship Games Division I FBS 2007 * 2008 * 2009 * 2010 * 2011 * 2012 *
2013 * 2014 * 2015 * 2016 * 2017 * 2018 * 2019 *
</item 8>

<item 9>
2018 College Football Playoff National Championship
Jump up ^ Gartland , Dan ( January 3 , 2018 ) . `` UCF Is Hanging a National Championship Banner at Its Stadium '' . Sports Illustrated . Retrieved January 10 ,
2018 . Jump up ^ Chavez , Chris ( January 3 , 2018 ) . `` UCF AD Says Coaches Will Be Paid National Championship Bonuses '' . SI.com . Retrieved January 11 ,
2018 . Jump up ^ September 1 , 2018 ( January 9 , 2018 ) . `` Alabama is No. 1 in final AP poll for 11th time ; UCF 6th '' . Msn.com . Retrieved January 16 ,
2018 . Jump up ^ Kercheval , Ben ( January 9 , 2018 ) . `` Coaches Poll Top 25 : Alabama unanimous No. 1 , UCF finishes outside top five '' . CBSSports.com .
Retrieved January 16 , 2018 . External links ( edit ) Media related to 2018 College Football Playoff National Championship at Wikimedia Commons College Football
Playoff Championships Overview College Football
</item 9>

<item 10>
2018 College Football Playoff National Championship
2018 College football Playoff national championship - wikipedia 2018 College football Playoff national championship Jump to : navigation , search 2018 College
Football Playoff National Championship Alabama Crimson Tide Georgia Bulldogs ( 12 -- 1 ) ( 13 -- 1 ) SEC SEC 26 23 Head coach : Nick Saban Head coach : Kirby
Smart AP Coaches CFP AP Coaches CFP OT Total Alabama 0 0 10 10 6 26 Georgia 0 13 7 0 23 Date January 8 , 2018 Season 2017 Stadium Mercedes - Benz Stadium
Location Atlanta , Georgia MVP Offensive : # 13 QB Tua Tagovailoa , Fr . Alabama Defensive : # 94 DT Da'Ron Payne , Jr .
</item 10>
=== END ITEM id=972efe7fc241bbac ===

=== ITEM id=37e3cfb4d2973736 ===
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
Additionally , director Cimino instructed the props department to fashion complete Pennsylvania IDs for each of them , including driver 's licenses , medical
cards , and various other pieces of paraphernalia , in order to enhance each actor 's sense of his character . Pre-production ( edit ) There has been
considerable debate , controversy , and conflicting stories about how The Deer Hunter was initially developed and written . Director and co-writer Michael
Cimino , writer Deric Washburn , and producers Barry Spikings and Michael Deeley all have different versions of how the film came to be . Development ( edit )
In 1968 , the record company EMI formed a new company called EMI Films , headed by producers Barry Spikings and Michael Deeley . Deeley purchased the first
draft of a spec script called The Man Who Came to Play , written by Louis Garfinkle and Quinn K. Redeker , for $19,000 . The spec script was about people who go
to Las Vegas to play Russian roulette . `` The screenplay had struck me as brilliant , '' wrote Deeley , `` but it was n't complete .
</item 1>

<item 2>
The Deer Hunter
I 'd been working 20 hours a day for a month . I got on the plane the next day , and I went back to Manhattan and my carpenter job . '' Deeley 's reaction to
the revised script ( edit ) Deeley felt the revised script , now called The Deer Hunter , broke fresh ground for the project . The protagonist in the Redeker /
Garfinkle script , Merle , was an individual who sustained a bad injury in active service and was damaged psychologically by his violent experiences , but was
nevertheless a tough character with strong nerves and guts . Cimino and Washburn 's revised script distilled the three aspects of Merle 's personality and
separated them out into three distinct characters . They became three old friends who grew up in the same small industrial town and worked in the same steel
mill , and in due course were drafted together to Vietnam . In the original script , the roles of Merle ( later renamed Mike ) and Nick were reversed in the
last half of the film . Nick returns home to Linda , while Mike remains in Vietnam , sends money home to help Steven , and meets his tragic fate at the Russian
roulette table .
</item 2>

<item 3>
The Deer Hunter
2 Washburn 's claim 3.2. 3 Deeley 's reaction to the revised script 4 Filming 4.1 The wedding scenes 4.2 The bar and the steel mill 4.3 Hunting the deer 4.4
Vietnam and the Russian roulette scenes 4.5 Filming locations 5 Post-production 5.1 Sound design 5.2 Previews 6 Soundtrack 6.1 Selected tracks 7 Release 8
Analysis 8.1 Controversy over Russian roulette 8.1. 1 Critics ' response 8.1.
</item 3>

<item 4>
The Deer Hunter
The Deer Hunter - wikipedia The Deer Hunter Jump to : navigation , search For other uses , see Deer Hunter ( disambiguation ) . The Deer Hunter UK theatrical
release poster Directed by Michael Cimino Produced by Barry Spikings Michael Deeley Michael Cimino John Peverall Screenplay by Deric Washburn Story by Deric
Washburn Michael Cimino Louis Garfinkle Quinn K. Redeker Starring Robert De Niro John Cazale John Savage Meryl Streep Christopher Walken Music by Stanley Myers
Cinematography Vilmos Zsigmond Edited by Peter Zinner Production company EMI Distributed by Universal Pictures Release date December 8 , 1978 ( 1978 - 12 - 08 )
( Los Angeles ) February 23 , 1979 ( 1979 - 02 - 23 ) ( United States ) Running time 183 minutes Country United States Language English Russian Vietnamese
French Budget $15 million Box office $49 million The Deer Hunter is a 1978 American epic war drama film co-written and directed by Michael Cimino about a trio
of Russian American steelworkers whose lives are changed forever after they fight in the Vietnam War .
</item 4>

<item 5>
The Deer Hunter
His direction to his actors : `` You put the gun to your head , Chris , you shoot , you fall over and Bobby cradles your head . '' Filming locations ( edit )
Lemko Hall St. Theodosius Russian Orthodox Cathedral , in the Tremont neighborhood of Cleveland , Ohio . The name plaque is clearly visible in one scene . Lemko
Hall , Cleveland , Ohio . Also located in Tremont , the wedding banquet was filmed here . The name is clearly visible in one scene . U.S. Steel Central Furnaces
in Cleveland , Ohio . Opening sequence steel mill scenes . Patpong , Bangkok , Thailand , the area used to represent Saigon 's red light district . Sai Yok ,
Kanchanaburi Province , Thailand Mount Baker - Snoqualmie National Forest and Nooksack Falls in the North Cascades range of , Washington , deer hunting scenes .
Steubenville , Ohio , for some mill and neighborhood shots . Struthers , Ohio , for external house and long - range road shots .
</item 5>

<item 6>
The Deer Hunter
As a result , Cimino was acting without the day - to - day supervision of a producer . Because Deeley was busy overseeing in the production of Sam Peckinpah 's
Convoy ( 1978 ) , he hired John Peverall to oversee Cimino 's shoot . Peverall 's expertise with budgeting and scheduling made him a natural successor to Relyea
, and Peverall knew enough about the picture to be elevated to producer status . `` John is a straightforward Cornishman who had worked his way up to become a
production supervisor , '' wrote Deeley , `` and we employed him as EMI 's watchman on certain pictures . '' The wedding scenes ( edit ) St. Theodosius Russian
Orthodox Cathedral in Cleveland , Ohio . Site of the wedding scene . The wedding scenes were filmed at the historic St. Theodosius Russian Orthodox Cathedral in
the Tremont neighborhood of Cleveland , Ohio . The wedding took five days to film . St. Theodosius ' Father Stephen Kopestonsky was cast as the priest at the
wedding . The reception scene was filmed at nearby Lemko Hall .
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
The Deer Hunter
The reception scene was filmed at nearby Lemko Hall . The amateur extras lined up for the crowded wedding - dance sequences drank real liquor and beer . The
scenes were filmed in the summer , but were set in the fall . To accomplish a look of fall , individual leaves were removed from deciduous trees . Zsigmond also
had to desaturate the colors of the exterior shots , partly in camera and in the laboratory processing . The production manager asked each of the Russian
immigrant extras to bring to the location a gift - wrapped box to double for wedding presents . The manager figured if the extras did this , not only would the
production save time and money , but the gifts would also look more authentic . Once the unit wrapped and the extras disappeared , the crew discovered to their
amusement that the boxes were n't empty but filled with real presents , from china to silverware . `` Who got to keep all these wonderful offerings , '' wrote
Deeley `` is a mystery I never quite fathomed . '' Cimino originally claimed that the wedding scene would take up 21 minutes of screen time . In the end , it
took 51 minutes .
</item 8>

<item 9>
The Deer Hunter
The cast and crew viewed large amounts of news footage from the war to ensure authenticity . The film was shot over a period of six months . The Clairton scenes
comprise footage shot in eight different towns in four states : West Virginia , Pennsylvania , Washington , and Ohio . The initial budget of the film was $8.5
million . Meryl Streep accepted the role of the `` vague , stock girlfriend '' , in order to remain for the duration of filming with John Cazale , who had been
diagnosed with lung cancer . De Niro had spotted Streep in her stage production of The Cherry Orchard and had suggested that she play his girlfriend Linda .
Before the beginning of principal photography , Deeley had a meeting with the film 's appointed line producer Robert Relyea . Deeley hired Relyea after meeting
him on the set of Bullitt ( 1968 ) and was impressed with his experience . However , Relyea declined the job , refusing to disclose his reason why . Deeley
suspected that Relyea sensed in director Cimino something that would have made production difficult .
</item 9>

<item 10>
The Deer Hunter
Struthers , Ohio , for external house and long - range road shots . Also including , the town 's bowling alley is the Bowladrome Lanes , located at 56 State
Street , Struthers , Ohio . Weirton , West Virginia , for mill and trailer shots . River Kwai , Thailand , Prison camp and initial Russian roulette scene .
Post-production ( edit ) By this point , The Deer Hunter had cost $13 million and the film still had to go through an arduous post-production . Film editor
Peter Zinner was given 600,000 feet of printed film to edit , a monumental task at the time . Producers Spikings and Deeley were pleased with the first cut ,
which ran for three and a half hours . `` We were thrilled by what we saw , '' wrote Deeley , `` and knew that within the three and a half hours we watched
there was a riveting film . '' Executives from Universal , including Lew Wasserman and Sid Sheinberg , were not very enthusiastic . `` I think they were shocked
, '' recalled Spikings . `` What really upset them was ' God Bless America ' .
</item 10>
=== END ITEM id=37e3cfb4d2973736 ===

=== ITEM id=d7354f99031e4c26 ===
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
In the end , it took 51 minutes . Deeley believes that Cimino always planned to make this prologue last for an hour , and `` the plan was to be advanced by
stealth rather than straight dealing . '' At this point in the production , nearly halfway through principal photography , Cimino was already over budget , and
producer Spikings could tell from the script that shooting the extended scene could sink the project . The bar and the steel mill ( edit ) The bar was specially
constructed in an empty storefront in Mingo Junction , Ohio for $25,000 ; it later became an actual saloon for local steel mill workers . U.S. Steel allowed
filming inside its Cleveland mill , including placing the actors around the furnace floor , only after securing a $5 million insurance policy . Other filming
took place in Pittsburgh . Hunting the Deer ( edit ) The first deer to be shot was depicted in a `` gruesome close - up '' , although he was hit with a
tranquilizer dart . The stag that Michael allows to get away later was the same one used on TV commercials for the Connecticut Life Insurance Company .
</item 1>

<item 2>
The Deer Hunter
The Deer Hunter - wikipedia The Deer Hunter Jump to : navigation , search For other uses , see Deer Hunter ( disambiguation ) . The Deer Hunter UK theatrical
release poster Directed by Michael Cimino Produced by Barry Spikings Michael Deeley Michael Cimino John Peverall Screenplay by Deric Washburn Story by Deric
Washburn Michael Cimino Louis Garfinkle Quinn K. Redeker Starring Robert De Niro John Cazale John Savage Meryl Streep Christopher Walken Music by Stanley Myers
Cinematography Vilmos Zsigmond Edited by Peter Zinner Production company EMI Distributed by Universal Pictures Release date December 8 , 1978 ( 1978 - 12 - 08 )
( Los Angeles ) February 23 , 1979 ( 1979 - 02 - 23 ) ( United States ) Running time 183 minutes Country United States Language English Russian Vietnamese
French Budget $15 million Box office $49 million The Deer Hunter is a 1978 American epic war drama film co-written and directed by Michael Cimino about a trio
of Russian American steelworkers whose lives are changed forever after they fight in the Vietnam War .
</item 2>

<item 3>
*The Deer Hunter* filming locations included Cleveland’s Tremont neighborhood: St. Theodosius Russian Orthodox Cathedral and Lemko Hall, used for church and
wedding scenes, plus U.S. Steel’s Central Furnaces for mill scenes. Patpong in Bangkok represented Saigon’s red-light district; Sai Yok, Thailand, and
Washington’s Mount Baker–Snoqualmie National Forest and Nooksack Falls provided other settings, including deer-hunting scenes. Steubenville and Struthers, Ohio
</item 3>

<item 4>
The sources document *The Deer Hunter* (1978)’s production, locations, editing, reception, and legacy. Filming took place in Ohio and West Virginia, Cleveland,
Bangkok and other parts of Thailand, and the Pacific Northwest. Editor Peter Zinner assembled 600,000 feet of footage; his 3½-hour cut was shortened. References
cover the film’s controversial Russian roulette scenes, reviews and scholarship, soundtrack and home-video releases, box office, and 1979 Oscars.
</item 4>

<item 5>
The Deer Hunter
His direction to his actors : `` You put the gun to your head , Chris , you shoot , you fall over and Bobby cradles your head . '' Filming locations ( edit )
Lemko Hall St. Theodosius Russian Orthodox Cathedral , in the Tremont neighborhood of Cleveland , Ohio . The name plaque is clearly visible in one scene . Lemko
Hall , Cleveland , Ohio . Also located in Tremont , the wedding banquet was filmed here . The name is clearly visible in one scene . U.S. Steel Central Furnaces
in Cleveland , Ohio . Opening sequence steel mill scenes . Patpong , Bangkok , Thailand , the area used to represent Saigon 's red light district . Sai Yok ,
Kanchanaburi Province , Thailand Mount Baker - Snoqualmie National Forest and Nooksack Falls in the North Cascades range of , Washington , deer hunting scenes .
Steubenville , Ohio , for some mill and neighborhood shots . Struthers , Ohio , for external house and long - range road shots .
</item 5>

<item 6>
The Deer Hunter
Additionally , director Cimino instructed the props department to fashion complete Pennsylvania IDs for each of them , including driver 's licenses , medical
cards , and various other pieces of paraphernalia , in order to enhance each actor 's sense of his character . Pre-production ( edit ) There has been
considerable debate , controversy , and conflicting stories about how The Deer Hunter was initially developed and written . Director and co-writer Michael
Cimino , writer Deric Washburn , and producers Barry Spikings and Michael Deeley all have different versions of how the film came to be . Development ( edit )
In 1968 , the record company EMI formed a new company called EMI Films , headed by producers Barry Spikings and Michael Deeley . Deeley purchased the first
draft of a spec script called The Man Who Came to Play , written by Louis Garfinkle and Quinn K. Redeker , for $19,000 . The spec script was about people who go
to Las Vegas to play Russian roulette . `` The screenplay had struck me as brilliant , '' wrote Deeley , `` but it was n't complete .
</item 6>

<item 7>
The Deer Hunter
Struthers , Ohio , for external house and long - range road shots . Also including , the town 's bowling alley is the Bowladrome Lanes , located at 56 State
Street , Struthers , Ohio . Weirton , West Virginia , for mill and trailer shots . River Kwai , Thailand , Prison camp and initial Russian roulette scene .
Post-production ( edit ) By this point , The Deer Hunter had cost $13 million and the film still had to go through an arduous post-production . Film editor
Peter Zinner was given 600,000 feet of printed film to edit , a monumental task at the time . Producers Spikings and Deeley were pleased with the first cut ,
which ran for three and a half hours . `` We were thrilled by what we saw , '' wrote Deeley , `` and knew that within the three and a half hours we watched
there was a riveting film . '' Executives from Universal , including Lew Wasserman and Sid Sheinberg , were not very enthusiastic . `` I think they were shocked
, '' recalled Spikings . `` What really upset them was ' God Bless America ' .
</item 7>

<item 8>
The Deer Hunter
The reception scene was filmed at nearby Lemko Hall . The amateur extras lined up for the crowded wedding - dance sequences drank real liquor and beer . The
scenes were filmed in the summer , but were set in the fall . To accomplish a look of fall , individual leaves were removed from deciduous trees . Zsigmond also
had to desaturate the colors of the exterior shots , partly in camera and in the laboratory processing . The production manager asked each of the Russian
immigrant extras to bring to the location a gift - wrapped box to double for wedding presents . The manager figured if the extras did this , not only would the
production save time and money , but the gifts would also look more authentic . Once the unit wrapped and the extras disappeared , the crew discovered to their
amusement that the boxes were n't empty but filled with real presents , from china to silverware . `` Who got to keep all these wonderful offerings , '' wrote
Deeley `` is a mystery I never quite fathomed . '' Cimino originally claimed that the wedding scene would take up 21 minutes of screen time . In the end , it
took 51 minutes .
</item 8>

<item 9>
The Deer Hunter was filmed over six months on an initial $8.5 million budget. To ensure authenticity, cast and crew studied extensive war footage. Clairton
scenes were filmed across eight towns in West Virginia, Pennsylvania, Washington, and Ohio. Meryl Streep took the role of Linda to stay with John Cazale, who
had lung cancer; Robert De Niro had suggested her after seeing her in The Cherry Orchard. Line producer Robert Relyea declined the job, and producer Deeley
</item 9>

<item 10>
The Deer Hunter
2 Washburn 's claim 3.2. 3 Deeley 's reaction to the revised script 4 Filming 4.1 The wedding scenes 4.2 The bar and the steel mill 4.3 Hunting the deer 4.4
Vietnam and the Russian roulette scenes 4.5 Filming locations 5 Post-production 5.1 Sound design 5.2 Previews 6 Soundtrack 6.1 Selected tracks 7 Release 8
Analysis 8.1 Controversy over Russian roulette 8.1. 1 Critics ' response 8.1.
</item 10>
=== END ITEM id=d7354f99031e4c26 ===
