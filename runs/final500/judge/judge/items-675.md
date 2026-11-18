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

=== ITEM id=89e72c2529280cc5 ===
Gold sentences and facts:
Sentence 0: `` Ca n't Get You Out of My Head '' is a song recorded by Australian singer Kylie Minogue for her eighth studio album , titled Fever , which she released in 2001 . The song was released in Australia by Parlophone as the lead single from the album on 8 September 2001 . It was released on 17 September 2001 in the United Kingdom . In the United States , the single was released on 18 February 2002 . Jointly written , composed , and produced by Cathy Dennis and Rob Davis , `` Ca n't Get You Out of My Head '' is a midtempo dance - pop song which lyrically details its narrator 's obsession towards her lover . The song is famous for its `` la la la '' hook .
  Fact 0.0: 'Can't Get You Out of My Head' is a song.
  Fact 0.1: 'Can't Get You Out of My Head' is recorded by Australian singer Kylie Minogue.
  Fact 0.2: 'Can't Get You Out of My Head' was recorded for Kylie Minogue's eighth studio album.
  Fact 0.3: The eighth studio album is titled Fever.
  Fact 0.4: Kylie Minogue released Fever in 2001.
  Fact 0.5: 'Can't Get You Out of My Head' was released in Australia by Parlophone.
  Fact 0.6: 'Can't Get You Out of My Head' was released in Australia as the lead single from Fever.
  Fact 0.7: The Australian release of 'Can't Get You Out of My Head' was on 8 September 2001.
  Fact 0.8: 'Can't Get You Out of My Head' was released in the United Kingdom on 17 September 2001.
  Fact 0.9: 'Can't Get You Out of My Head' was released in the United States as a single on 18 February 2002.
  Fact 0.10: 'Can't Get You Out of My Head' was jointly written by Cathy Dennis and Rob Davis.
  Fact 0.11: 'Can't Get You Out of My Head' was jointly composed by Cathy Dennis and Rob Davis.
  Fact 0.12: 'Can't Get You Out of My Head' was jointly produced by Cathy Dennis and Rob Davis.
  Fact 0.13: 'Can't Get You Out of My Head' is a midtempo dance-pop song.
  Fact 0.14: 'Can't Get You Out of My Head' lyrically details its narrator's obsession towards her lover.
  Fact 0.15: The song is famous for its 'la la la' hook.

Retrieved items:
<item 1>
Can't Get You Out of My Head
My A&R at the time , Miles Leonard and Jamie Nelson , said , `` We 've got something . Come into the office . We 've got to play you something . '' So I went to
the EMI office , and I had the same reaction you did , about 20 seconds in . I could n't even fathom what I was hearing . It just ... did something . I was
beside myself . Then at the end of the song , panic set in . I was saying `` Are you sure we 've got this song ? Do n't tell me that we do n't ! Is it secured ?
Can we have it ? '' And we did ! And that kick - started a whole different phase in my career . -- Minogue , about her reaction to hearing the demo of the song
. `` Ca n't Get You Out of My Head '' was jointly written , composed , and produced by Cathy Dennis and Rob Davis . Dennis and Davis had been brought together
by British artist manager Simon Fuller , who wanted the duo to come up with a song for British pop group S Club 7 .
</item 1>

<item 2>
Can't Get You Out of My Head
Single by Kylie Minogue from the album Fever B - side `` Boy '' `` Rendezvous at Sunset '' Released 8 September 2001 ( 2001 - 09 - 08 ) Format CD digital
download Recorded 2001 ; London , England Genre Synthpop dance - pop Length 3 : 50 Label Parlophone Songwriter ( s ) Cathy Dennis Rob Davis Producer ( s ) Cathy
Dennis Rob Davis Kylie Minogue singles chronology `` Your Disco Needs You '' ( 2001 ) `` Ca n't Get You Out of My Head '' ( 2001 ) `` In Your Eyes '' ( 2002 )
`` Your Disco Needs You '' ( 2001 ) `` Ca n't Get You Out of My Head '' ( 2001 ) `` In Your Eyes '' ( 2002 ) Music video `` Ca n't Get You Out of My Head '' on
YouTube `` Ca n't Get You Out of My Head '' is a song recorded by Australian singer Kylie Minogue for her eighth studio album , titled Fever , which she
released in 2001 . The song was released in Australia by Parlophone as the lead single from the album on 8 September 2001 .
</item 2>

<item 3>
Can't Get You Out of My Head
The music , excluding the guitar part , was programmed using Korg Triton workstation via MIDI . Dennis later remarked : `` Even though Kylie was n't the first
artist to be offered the song , I do n't believe it was meant to go to anyone other than Kylie , and I do n't believe anyone else would have done the incredible
job she did with it , with the video , looking super-hot ! '' Composition ( edit ) `` Ca n't Get You Out of My Head '' A 21 - second sample of `` Ca n't Get You
Out of My Head '' containing the hook and part of the chorus of the song . The song is known for its `` la la la '' hook . Problems playing this file ? See
media help . `` Ca n't Get You Out of My Head '' is a `` robotic '' midtempo dance - pop song , with a tempo of 126 beats per minute .
</item 3>

<item 4>
Can't Get You Out of My Head
Ca n't Get You Out of My Head - Wikipedia Ca n't Get You Out of My Head Jump to : navigation , search `` Ca n't Get You Out of My Head '' Single cover used for
CD # 02 and digital releases of the song .
</item 4>

<item 5>
Can't Get You Out of My Head
( 2nd ed . ) ISBN 1 - 905287 - 89 - 5 External links ( edit ) A page on `` Ca n't Get You Out of My Head '' at Kylie Minogue 's official website Lyrics of this
song at MetroLyrics Kylie Minogue songs Singles discography Kylie `` I Should Be So Lucky '' `` The Loco - Motion '' `` Je ne sais pas pourquoi '' `` It 's No
Secret '' `` Got to Be Certain '' `` Turn It into Love '' Enjoy Yourself `` Hand on Your Heart '' `` Would n't Change a Thing '' `` Never Too Late '' `` Tears
on My Pillow '' `` Especially for You '' Rhythm of Love `` Better the Devil You Know '' `` Step Back in Time '' `` What Do I Have to Do '' `` Shocked '' Let 's
Get to It `` Word Is Out '' `` Give Me Just a Little More Time '' `` Finer Feelings '' `` If You Were with Me Now '' `` I Guess I Like It Like That '' Greatest
Hits `` What Kind of Fool ( Heard All That Before ) '' `` Celebration '' Kylie Minogue `` Confide in Me '' `` Put Yourself in My Place '' `` Where Is the
Feeling ? ''
</item 5>

<item 6>
Can't Get You Out of My Head
It was released on 17 September 2001 in the United Kingdom . In the United States , the single was released on 18 February 2002 . Jointly written , composed ,
and produced by Cathy Dennis and Rob Davis , `` Ca n't Get You Out of My Head '' is a midtempo dance - pop song which lyrically details its narrator 's
obsession towards her lover . The song is famous for its `` la la la '' hook . In addition to positive reviews from music critics , `` Ca n't Get You Out of My
Head '' found commercial success on a large scale . It peaked at number one on the charts of Austria , Belgium , France , Germany , Italy , Poland , Switzerland
, the United Kingdom , and every other European country excluding Finland . It also topped the charts of Minogue 's native country Australia , Canada and New
Zealand . In the United States , the song peaked at number seven on the US Billboard Hot 100 chart , becoming Minogue 's biggest hit in the region since `` The
Loco - Motion '' . `` Ca n't Get You Out of My Head '' reportedly reached number one in 40 countries across the globe .
</item 6>

<item 7>
Can't Get You Out of My Head
The song was recorded using Cubase music software , which Davis ran on his Mac computer . Davis began playing an acoustic guitar and ran a 125 beats per minute
drum loop , on which Dennis , herself a singer who had enjoyed chart success in the United States , began singing the line `` I just ca n't get you out of my
head '' in the key of D minor . After three and a half hours , the demo was recorded and the vocals were laid afterwards . Dennis called their recording setup
for the song `` the most primitive set - up you could imagine ! Different producers work in different ways . But it 's good to be reminded you do n't have to be
reliant on equipment . A song is about melody and lyrics and being able to take something away in your memory that is going to haunt you . '' She also regarded
their production as a `` very natural and fluid process , '' saying : `` We know how hard we work sometimes to write songs and then spend months picking them to
pieces , but this was the easiest process , the chemicals were all happy and working together . ''
</item 7>

<item 8>
Can't Get You Out of My Head
Through the lyrics of the song , its singer expresses an obsession with an anonymous figure . Dorian Lynskey from The Guardian termed the song a `` mystery ''
as Minogue never reveals the identity of her object of infatuation . The critic suggested that the person Minogue is referring to is either `` a partner , an
evasive one - night stand or someone who does n't know she exists . '' Writing for the same newspaper , Everett True identified a `` darker element '' in the
simple lyrics and felt this sentiment was echoed in Minogue 's restrained vocals . Further , True emphasised that while Minogue 's 1987 single `` I Should Be So
Lucky '' had presented an optimistic romantic future , `` Ca n't Get You Out of My Head '' focuses on an `` unhealthy '' and potentially destructive obsession .
He also noted that in the former song , Minogue played `` the wide - eyed ingénue with alacrity , '' but in the latter song she is aware of the harmful nature
of her infatuation , calling it a `` desire that is wholly dependent on her own self - control . ''
</item 8>

<item 9>
Can't Get You Out of My Head
: 26 Digital download `` Ca n't Get You Out of My Head '' -- 3 : 52 `` Boy '' -- 3 : 48 Charts ( edit ) Weekly charts ( edit ) Chart ( 2001 -- 02 ) Peak
position Australia ( ARIA ) Austria ( Ö3 Austria Top 40 ) Belgium ( Ultratop 50 Flanders ) Belgium ( Ultratop 50 Wallonia ) Brazil ( ABPD ) Canada ( Canadian
Hot 100 ) Denmark ( Tracklisten ) Finland ( Suomen virallinen lista ) 5 France ( SNEP ) Germany ( Official German Charts ) Greece ( IFPI Greece ) Ireland ( IRMA
) Italy ( FIMI ) Netherlands ( Single Top 100 ) New Zealand ( Recorded Music NZ ) Norway ( VG - lista ) Poland ( Polish Singles Chart ) Portugal ( Billboard )
Romania ( Romanian Top 100 ) Scotland ( Official Charts Company ) South Africa ( RiSA ) Spain ( PROMUSICAE ) Sweden ( Sverigetopplistan ) Switzerland (
Schweizer Hitparade ) UK Singles ( Official Charts Company ) US Billboard Hot 100 7 US Adult Top 40 ( Billboard ) 23 US Dance Club Songs ( Billboard )
</item 9>

<item 10>
Can't Get You Out of My Head
In late 2012 , `` Ca n't Get You Out of My Head '' was re-recorded by Minogue for inclusion in her orchestral compilation album , The Abbey Road Sessions . On
the album , Minogue reworked 16 of her past songs with an orchestra , which , according to Nick Levine from BBC Music , `` re-imagine them without the disco
glitz and vocal effects . '' The Abbey Road Sessions - version of `` Ca n't Get You Out of My Head '' features a `` more dramatic , fully fleshed out '' musical
arrangement , and follows a pizzicato playing technique , in which the strings of a string instrument are continuously plucked . Critical reception ( edit )
Minogue performing `` Ca n't Get You Out of My Head '' during her 2009 North American For You , For Me Tour . The song received overwhelming critical acclaim ,
subsequently being cited as one of the greatest songs of the 2000s . Chris True from AllMusic picked the song as a highlight of Fever , commenting that it ``
pulses and grooves like no other she 's ( Minogue 's ) recorded . ''
</item 10>
=== END ITEM id=89e72c2529280cc5 ===

=== ITEM id=5480660a21a65238 ===
Gold sentences and facts:
Sentence 0: The Uralic languages ( / jʊəˈrælɪk / ; sometimes called Uralian languages / jʊəˈreɪliən / ) form a language family of 38 languages spoken by approximately 25 million people , predominantly in Northern Eurasia . The Uralic languages with the most native speakers are Hungarian , Finnish , and Estonian , which are the official languages of Hungary , Finland , and Estonia , respectively , and of the European Union . Other Uralic languages with significant numbers of speakers are Erzya , Moksha , Mari , Udmurt , and Komi , which are officially recognized languages in various regions of Russia .
  Fact 0.0: The Uralic languages form a language family.
  Fact 0.1: The Uralic languages are sometimes called Uralian languages.
  Fact 0.2: The Uralic language family has 38 languages.
  Fact 0.3: The Uralic languages are spoken by approximately 25 million people.
  Fact 0.4: The Uralic languages are spoken predominantly in Northern Eurasia.
  Fact 0.5: Hungarian, Finnish, and Estonian are the Uralic languages with the most native speakers.
  Fact 0.6: Hungarian is the official language of Hungary.
  Fact 0.7: Finnish is the official language of Finland.
  Fact 0.8: Estonian is the official language of Estonia.
  Fact 0.9: Hungarian, Finnish, and Estonian are official languages of the European Union.
  Fact 0.10: Erzya, Moksha, Mari, Udmurt, and Komi are other Uralic languages with significant numbers of speakers.
  Fact 0.11: Erzya, Moksha, Mari, Udmurt, and Komi are officially recognized languages in various regions of Russia.

Retrieved items:
<item 1>
Uralic languages
Mutual intelligibility ( edit ) The Estonian philologist Mall Hellam proposed cognate sentences that she asserted to be mutually intelligible among the three
most widely spoken Uralic languages : Finnish , Estonian , and Hungarian : Estonian : Elav kala ujub vee all . Finnish : Elävä kala ui veden alla . Hungarian :
Eleven hal úszik a víz alatt . English : A live fish is swimming underwater . However , linguist Geoffrey Pullum reports that neither Finns nor Hungarians could
understand the other language 's version of the sentence . Comparison ( edit ) No Uralic language has exactly the idealized typological profile of the family .
</item 1>

<item 2>
Uralic languages
and refined in Viitso ( 2000 ) : Finno - Ugric Saamic -- Fennic ( consonant gradation ) Saamic Fennic Eastern Finno - Ugric Mordva ( node ) Mari Permian --
Ugric ( * δ > * l ) Permian Ugric ( * s * š * ś > * ɬ * ɬ * s ) Hungarian Khanty Mansi The grouping of the four bottom - level branches remains to some degree
open to interpretation , with competing models of Finno - Saamic vs. Eastern Finno - Ugric ( Mari , Mordvinic , Permic - Ugric ; * k > ɣ between vowels ,
degemination of stops ) and Finno - Volgaic ( Finno - Saamic , Mari , Mordvinic ; * δ́ > δ between vowels ) vs. Permic - Ugric . Viitso finds no evidence for a
Finno - Permic grouping .
</item 2>

<item 3>
Uralic languages
Classification ( edit ) Relative numbers of speakers of Uralic languages Hungarian 56 % Finnish 20 % Estonian 4.2 % Erzya 2.8 % Moksha 2.5 % Mari 2 % Udmurt 1.9
% Komi 1.6 % Other 8.9 % The Uralic family comprises nine undisputed groups with no consensus classification between them . ( Some of the proposals are listed
in the next section . ) An agnostic approach treats them as separate branches . Obsolete or native names are displayed in italics .
</item 3>

<item 4>
Uralic languages
Uralic languages - wikipedia Uralic languages Jump to : navigation , search Not to be confused with Urali language . This article needs additional citations for
verification . Please help improve this article by adding citations to reliable sources . Unsourced material may be challenged and removed . ( July 2017 ) (
Learn how and when to remove this template message ) Uralic Uralian languages Geographic distribution Central , Eastern , and Northern Europe , North Asia
Linguistic classification One of the world 's primary language families Proto - language Proto - Uralic Subdivisions Finnic Hungarian Khanty Mansi Mari
Mordvinic Permic Sami Samoyedic ISO 639 - 5 urj Glottolog ural1272 Geographical distribution of the Uralic languages The Uralic languages ( / jʊəˈrælɪk / ;
sometimes called Uralian languages / jʊəˈreɪliən / ) form a language family of 38 languages spoken by approximately 25 million people , predominantly in
Northern Eurasia .
</item 4>

<item 5>
Uralic languages
The Uralic languages with the most native speakers are Hungarian , Finnish , and Estonian , which are the official languages of Hungary , Finland , and Estonia
, respectively , and of the European Union . Other Uralic languages with significant numbers of speakers are Erzya , Moksha , Mari , Udmurt , and Komi , which
are officially recognized languages in various regions of Russia . The name `` Uralic '' derives from the fact that the areas where the languages are spoken are
found on both sides of the Ural Mountains . Also , the original homeland ( Urheimat ) is commonly hypothesized to be in the vicinity of the Urals . Finno -
Ugric is sometimes used as a synonym for Uralic , though Finno - Ugric is widely understood to exclude the Samoyedic languages . Scholars who do not accept the
traditional notion that Samoyedic split first from the rest of the Uralic family may treat the terms as synonymous .
</item 5>

<item 6>
Uralic languages
Uralic languages Finnic Estonian Northeastern coastal South Estonian Võro Seto Finnish Kven Meänkieli Ingrian Karelian Karelian proper Livvi Ludic Livonian †
Veps Votic Sami Akkala † Inari Kainuu † Kemi † Kildin Lule Northern Pite Skolt Southern Ter Ume Mordvinic Erzya Moksha Mari Hill Mari Meadow Mari Permic Komi
Zyrian Permyak Yodzyak Udmurt Ugric Hungarian Khanty Mansi Samoyedic Enets Yurats † Nenets Forest Nenets Tundra Nenets Nganasan Kamassian † Mator † Selkup
Others Merya † Meshcherian † Muromian † † indicate extinct languages List of primary language families Africa Afro - Asiatic Austronesian Khoe Kx'a Niger --
Congo Nilo - Saharan ? Tuu Mande ? Songhay ? Ijaw ? Ubangian ? Kadu ? Isolates Bangime Hadza Jalaa Sandawe Kwadi ? Laal ?
</item 6>

<item 7>
Uralic languages
All of these hypotheses are minority views at the present time in Uralic studies . Other comparisons ( edit ) See also : Alternative theories of the Hungarian
language relations Various unorthodox comparisons have been advanced such as Finno - Basque , Hungarian - Etruscan , and Cal - Ugrian . These are considered at
best spurious fringe - theories by specialists . See also ( edit ) Finland portal Hungary portal Languages portal Uralic peoples Finno - Ugric peoples Notes (
edit ) Jump up ^ Hammarström , Harald ; Forkel , Robert ; Haspelmath , Martin , eds. ( 2017 ) . `` Uralic '' . Glottolog 3.0 . Jena , Germany : Max Planck
Institute for the Science of Human History . Jump up ^ Simons , Gary F. ; Fenning , Charles F. `` Uralic '' . Ethnologue . SIL International . Retrieved 14
February 2018 . Jump up ^ Tommola , Hannu ( 2010 ) . `` Finnish among the Finno - Ugrian languages '' . Mood in the Languages of Europe .
</item 7>

<item 8>
Uralic languages
Obsolete or native names are displayed in italics . Finnic ( Fennic , Baltic Finnic , Balto - Finnic , Balto - Fennic ) Hungarian ( Magyar ) Khanty ( Ostyak ,
Handi , Hantõ ) Mansi ( Vogul ) Mari ( Cheremis ) Mordvinic ( Mordvin , Mordvinian ) Permic ( Permian ) Sami ( Saami , Samic , Saamic , Lappic , Lappish )
Samoyedic ( Samoyed ) There is also historical evidence of a number of extinct languages of uncertain affiliation : Merya Muromian Meshcherian ( until 16th
century ? ) Traces of Finno - Ugric substrata , especially in toponymy , in the northern part of European Russia have been proposed as evidence for even more
extinct Uralic languages . Traditional classification ( edit ) All Uralic languages are thought to have descended , through independent processes of language
change , from Proto - Uralic . The internal structure of the Uralic family has been debated since the family was first proposed .
</item 8>

<item 9>
Uralic languages
The internal structure of the Uralic family has been debated since the family was first proposed . Doubts about the validity of most of the proposed higher -
order branchings ( grouping the nine undisputed families ) are becoming more common . A traditional classification of the Uralic languages has existed since the
late 19th century . It has enjoyed frequent adaptation in whole or in part in encyclopedias , handbooks , and overviews of the Uralic family . Donner 's model
is as follows : Ugric ( Ugrian ) Hungarian Ob - Ugric ( Ob Ugrian ) Khanty Mansi Finno - Permic ( Permian - Finnic ) Permic Finno - Volgaic ( Finno - Cheremisic
, Finno - Mari ) Volga - Finnic Mari Mordvinic Finno - Lappic ( Finno - Saamic , Finno - Samic ) Sami Finnic At Donner 's time , the Samoyedic languages were
still poorly known , and he was not able to address their position .
</item 9>

<item 10>
Uralic languages
The Finno - Permic grouping still holds some support , though the arrangement of its subgroups is a matter of some dispute . Mordvinic is commonly seen as
particularly closely related to or part of Finno - Samic . The term Volgaic ( or Volga - Finnic ) was used to denote a branch previously believed to include
Mari , Mordvinic and a number of the extinct languages , but it is now obsolete and considered a geographic classification rather than a linguistic one . Within
Ugric , uniting Mansi with Hungarian rather than Khanty has been a competing hypothesis to Ob - Ugric . Lexical isoglosses ( edit ) Lexicostatistics has been
used in defense of the traditional family tree . A recent re-evaluation of the evidence however fails to find support for Finno - Ugric and Ugric , suggesting
four lexically distinct branches ( Finno - Permic , Hungarian , Ob - Ugric and Samoyedic ) . One alternate proposal for a family tree , with emphasis on the
development of numerals , is as follows : Uralic ( * kektä `` 2 '' ,
</item 10>
=== END ITEM id=5480660a21a65238 ===

=== ITEM id=7b631f0ea33280a0 ===
Gold sentences and facts:
Sentence 0: Originally introduced as a classical tax system , in which companies were subject to tax on their profits and companies ' shareholders were also liable to income tax on the dividends that they received , the first major amendment to corporation tax saw it move to a dividend imputation system in 1973 , under which an individual receiving a dividend became entitled to an income tax credit representing the corporation tax already paid by the company paying the dividend . The classical system was reintroduced in 1999 , with the abolition of advance corporation tax and of repayable dividend tax credits . Another change saw the single main rate of tax split into three . Tax competition between jurisdictions reduced the main corporate tax rate from 28 % in 2008 - 2010 to a flat rate of 20 % as of April 2015 .
  Fact 0.0: Corporation tax was originally introduced as a classical tax system.
  Fact 0.1: Under the classical system, companies were subject to tax on their profits.
  Fact 0.2: Under the classical system, companies' shareholders were also liable to income tax on the dividends they received.
  Fact 0.3: The first major amendment to corporation tax moved it to a dividend imputation system in 1973.
  Fact 0.4: Under the 1973 dividend imputation system, an individual receiving a dividend became entitled to an income tax credit representing the corporation tax already paid by the company paying the dividend.
  Fact 0.5: The classical system was reintroduced in 1999.
  Fact 0.6: The 1999 reintroduction of the classical system came with the abolition of advance corporation tax.
  Fact 0.7: The 1999 reintroduction of the classical system came with the abolition of repayable dividend tax credits.
  Fact 0.8: Another change saw the single main rate of tax split into three.
  Fact 0.9: Tax competition between jurisdictions reduced the main corporate tax rate from 28% in 2008-2010.
  Fact 0.10: Tax competition between jurisdictions reduced the main corporate tax rate to a flat rate of 20% as of April 2015.

Retrieved items:
<item 1>
United Kingdom corporation tax
This ensured that where a company paid below the small companies ' rate ( 19 % in 2004 ) , dividend payments made to non-corporates ( for example , individuals
, trusts and personal representatives of deceased persons ) would be subject to additional corporation tax , bringing the corporation tax paid up to 19 % . For
example , a company making £ 10,000 profit , and making a £ 6,000 dividend distribution to an individual and £ 4,000 to another company would pay 19 %
corporation tax on the £ 6,000 . Although this measure substantially reduced the number of small businesses incorporating , the Chancellor in the 2006 Budget
said tax avoidance by small businesses through incorporation was still a major issue , and scrapped the starting rate entirely . Historic tax revenues ( edit )
UK Corporation Tax Revenue 1999 to 2016 10,000 20,000 30,000 40,000 50,000 Taxable profits and accounting profits ( edit ) The starting point for computing
taxable profits is profits before tax ( except for a life assurance company ) .
</item 1>

<item 2>
United Kingdom corporation tax
Financial year FY17 started on 1 April 2017 and ends on 31 March 2018 . Where a company 's accounting period straddles a financial year in which the corporation
tax rate has changed , the company 's profits for that period are split . For example , a company paying small companies ' rate with its accounting period
running from 1 January to 31 December , and making £ 100,000 of profit in 2007 , would be deemed to have made 90 / 365 * £ 100,000 = £ 24,657.53 in FY06 ( there
are 90 days between 1 January and 31 March ) , and 275 / 365 * £ 100,000 = £ 75 , 34.47 in FY07 , and would pay 19 % on the FY06 portion , and 20 % on the FY07
portion . From 1 April 2010 HM Revenue & Customs updated their terminology and the former Small Companies ' Rate is now called Small Profits Rate .
</item 2>

<item 3>
United Kingdom corporation tax
Corporate Tax rates for Financial Years 2008 -- 2014 2008 -- 2010 2011 2012 2013 2014 Small profits rate 21 % 20 % 20 % 20 % 20 % Small profits upper limit £
300,000 £ 300,000 £ 300,000 £ 300,000 £ 300,000 Marginal relief limits £ 300,001 -- £ 1,500,000 £ 300,001 -- £ 1,500,000 £ 300,001 -- £ 1,500,000 £ 300,001 -- £
1,500,000 £ 300,001 -- £ 1,500,000 Main rate 28 % 26 % 24 % 23 % 21 % Notes : The bands shown on the right hand side are divided by one plus the number of
associates ( usually the only associates a company has are fellow group companies , but the term is more widely defined ) The reduced rates do not apply to
close investment holding companies ( companies controlled by fewer than 5 people ( plus associates ) or by their directors / managers , whose main activity is
the holding of investments ) .
</item 3>

<item 4>
United Kingdom corporation tax
The 2002 Budget cut the starting rate to zero , with marginal relief applying in the same way . This caused a significant increase in the number of companies
being incorporated , as businesses that had operated as self - employed , paying income tax on profits from just over £ 5,000 , were attracted to the
corporation tax rate of 0 % on income up to £ 10,000 . Previously self - employed individuals could now distribute profits as dividend payments rather than
salaries . For companies with profits under £ 50,000 the corporation tax rate varied between 0 % and 19 % . Because dividend payments come with a basic rate tax
credit , provided the recipient did not earn more than the basic rate allowance , no further tax would be paid . The number of new companies being formed in
2002 -- 2003 reached 325,900 , an increase of 45 % on 2001 -- 2002 . The fact that individuals operating in this manner could potentially pay no tax at all was
felt by the government to be unfair tax avoidance , and the 2004 Budget introduced a Non-Corporate Distribution Rate .
</item 4>

<item 5>
United Kingdom corporation tax
Another change saw the single main rate of tax split into three . Tax competition between jurisdictions reduced the main corporate tax rate from 28 % in 2008 -
2010 to a flat rate of 20 % as of April 2015 . The UK government faced problems with its corporate tax structure , including European Court of Justice
judgements that aspects of it are incompatible with EU treaties . Tax avoidance schemes marketed by the financial sector have also proven an irritant , and been
countered by complicated anti-avoidance legislation . The complexity of the corporation tax system is a recognised issue . The Labour government , supported by
the Opposition parties , carried through wide - scale reform from the Tax Law Rewrite project , resulting in the Corporation Tax Act 2010 . The tax has slowly
been integrating generally accepted accounting practice , with the corporation tax system in various specific areas based directly on the accounting treatment .
Total net corporation tax receipts in 2014 - 15 were £ 43.0 billion , which was an increase of 7 % from £ 40.3 billion in 2013 - 14 .
</item 5>

<item 6>
United Kingdom corporation tax
Nor do they apply to companies in liquidation after the first 12 months . Authorised unit trusts and open - ended investment companies are taxed at the basic
rate of income tax which is 20 % as of 2010 Life assurance companies are taxed using the above rates on shareholder profits and 20 % on policy holder profits
Companies active in the oil and gas extraction industry in the UK or on the UK Continental Shelf are subject to an additional 10 % charge on their profits from
those activities Most companies are required to pay tax nine months and a day after the end of an accounting period . Larger companies are required to pay
quarterly instalments , in the seventh , tenth , thirteenth and sixteenth months after a full accounting period starts . These times are modified where an
accounting period lasts for less than twelve months . From 2005 onwards , for tax payable on oil and gas extraction profits , the third and fourth quarterly
instalments are merged , including the supplementary 10 % charge . In the financial year 2004 -- 2005 , approximately 39,000 companies paid corporation tax at
the main rate .
</item 6>

<item 7>
United Kingdom corporation tax
These 4.7 % of active companies are responsible for 75 % of all corporation tax receipts . Around 224,000 companies paid the small companies rate , with 34,000
benefiting from marginal relief . 264,000 were in the starting rate , with 269,000 benefiting from the lower band of marginal relief . The total revenue was £
41.9 bn from 831,885 companies . Only 23480 companies had a liability in excess of £ 100,000 . HM Revenue and Customs powers of enquiry ( edit ) HMRC has one
year from the normal filing date , which is itself one year after the end of the period of account , to open an enquiry into the return . This period is
extended if the return is filed late . The enquiry continues until all issues that HMRC wish to enquire about a return are dealt with . However , a company can
appeal to the Commissioners of Income Tax to close an enquiry if they feel there is undue delay . If either side disputes the amount of tax that is payable ,
they may appeal to either the General or Special Commissioners of Income Tax .
</item 7>

<item 8>
United Kingdom corporation tax
United Kingdom corporation tax - wikipedia United Kingdom corporation tax Jump to : navigation , search Throughout this article , the unqualified term `` pound
'' and the £ symbol refer to the Pound sterling . UK Corporation Tax Receipts from 1999 - 2016 , both in absolute terms and as % of GDP . Taxation in the United
Kingdom UK Government Departments HM Treasury HM Revenue and Customs UK Government VAT Income tax National Insurance Corporation tax Capital gains tax Motoring
taxes Inheritance tax Stamp duty Insurance Premium Tax Air Passenger Duty PAYE Scottish Government Revenue Scotland Scottish income tax Land and Buildings
Transaction Tax Scottish Landfill Tax Air Departure Tax Welsh Government Welsh Revenue Authority Land Transaction Tax Landfill Disposals Tax Local Government
Council Tax Domestic rates in Northern Ireland Business rates in England Business rates in Wales Business rates in Scotland Business rates in Northern Ireland
In the United Kingdom , corporation tax is a corporate tax levied in the United Kingdom on the profits made by UK - resident companies and on the profits of
entities registered overseas with permanent establishments in the UK . Until 1 April 1965 , companies were taxed at the same income tax rates as individual
taxpayers , with an additional profits tax levied on companies .
</item 8>

<item 9>
United Kingdom corporation tax
The 1979 Conservative Budget of Geoffrey Howe cut the small companies ' rate to 40 % , followed by a further cut in the 1982 Budget to 38 % . The Budgets of
1983 -- 1988 saw sharp cuts in both main and small companies ' rates , falling to 35 % and 25 % respectively . Budgets between 1988 and 2001 brought further
falls to a 30 % main rate and 19 % small companies ' rates . From April 1983 to March 1997 the small companies ' rate was pegged to the basic rate of income tax
. During the 1980s there was briefly a higher rate of tax imposed for capital profits . Chancellor Gordon Brown 's 1999 Budget introduced a 10 % starting rate
for profits from £ 0 to £ 10,000 , effective from April 2000 . Marginal relief applied meaning companies with profits of between £ 10,000 and £ 50,000 paid a
rate between the starting rate and the small companies ' rate ( 19 % in 2000 ) .
</item 9>

<item 10>
United Kingdom corporation tax
Relief for expenses ( edit ) Most direct expenses are deductible when calculating taxable income and chargeable gains . Notable exceptions include any costs of
entertaining clients . Companies with investment business may deduct certain indirect expenses known as `` expenses of management '' when calculating their
taxable profits . A similar relief is available for expenses of a life assurance company taxed on the I minus E basis which relate to the company 's basic life
assurance and general annuity business . Donations made to charities are also normally deducted in calculating taxable income if made under Gift Aid . Rates and
payment ( edit ) The 2007 Budget announced a main rate cut from 30 % to 28 % , effective from April 2008 . At the same time , the small companies ' rate was
increased from 19 % to 20 % from April 2007 , 21 % in April 2008 , to stop `` individuals artificially incorporating as small companies to avoid paying their
due share of tax , a practice if left unaddressed would cost the rest of the taxpaying population billions of pounds '' . The rate of corporation tax is
determined by the financial year , which runs from 1 April to the following 31 March .
</item 10>
=== END ITEM id=7b631f0ea33280a0 ===

=== ITEM id=368bbe382ce50fdc ===
Gold sentences and facts:
Sentence 0: Vaugier appeared in Need for Speed : Carbon as Nikki . She is the girlfriend of the antagonist in the game ( love interest of the protagonist ) .
  Fact 0.0: Vaugier appeared in Need for Speed: Carbon as Nikki.
  Fact 0.1: Nikki is the girlfriend of the antagonist in Need for Speed: Carbon.
  Fact 0.2: Nikki is the love interest of the protagonist in Need for Speed: Carbon.
Sentence 1: Year Title Role Notes 2006 Need for Speed : Carbon Nikki Voice , Mo - Cap , & Likeness 2015 Dying Light Jade Mo - Cap 2015 Hellraid Maine , The Female Kin Voice
  Fact 1.0: The filmography table lists a 2006 role as Nikki in Need for Speed: Carbon, with notes Voice, Mo-Cap, & Likeness.
  Fact 1.1: The filmography table lists a 2015 role as Jade in Dying Light, with notes Mo-Cap.
  Fact 1.2: The filmography table lists a 2015 role as Maine, The Female Kin in Hellraid, with notes Voice.

Retrieved items:
<item 1>
Emmanuelle Vaugier
She appeared in the film 40 Days and 40 Nights , which starred Josh Hartnett . She had starring roles in Suddenly Naked , in which she plays a Latin pop
sensation ; the TV - movie Mindstorm , a science - fiction thriller ; and Ripper , a psychological thriller about students who mysteriously disappear after
enrolling in a class about serial killers . She stars also in the sequel to Mirrors , which is directed by Victor Garcia . Maxim magazine featured her on the
cover of the February 2006 issue in the United States . Three months later , she landed spot # 31 on their annual Hot 100 list . Games ( edit ) Vaugier appeared
in Need for Speed : Carbon as Nikki . She is the girlfriend of the antagonist in the game ( love interest of the protagonist ) .
</item 1>

<item 2>
8 Mile (film)
Cast ( edit ) Eminem as Jimmy `` B - Rabbit '' Smith , Jr . Mekhi Phifer as Future , Jimmy 's closest friend and the rap battle host Brittany Murphy as Alex
Latourno , Jimmy 's love interest Kim Basinger as Stephanie , Jimmy and Lily 's mother Anthony Mackie as Papa Doc Michael Shannon as Greg Buehl , Stephanie 's
boyfriend Eugene Byrd as Wink Evan Jones as Cheddar Bob , Jimmy 's friend Omar Benson Miller as Sol George , Jimmy 's friend and Iz 's brother De'Angelo Wilson
as DJ Iz , Jimmy 's friend and Sol 's brother Taryn Manning as Janeane , Jimmy 's ex-girlfriend Proof as Lil ' Tic Xzibit as Mike ( Male Lunch Truck Rapper )
Craig Chandler as Paul Obie Trice as Male parking lot rapper Chloe Greenfield as Lily , Jimmy 's sister John Singleton as Bouncer Miz Korona as Vanessa ( Female
Lunch Truck Rapper ) Brandon T. Jackson ( uncredited ) as a Chin Tiki club - foe Music ( edit ) Main article : Music from and Inspired by the Motion Picture 8
Mile Music from and Inspired by the Motion Picture 8 Mile is the soundtrack to 8 Mile .
</item 2>

<item 3>
Emmanuelle Vaugier
Alexandra `` Nightingale '' Morgan 2005 Saw II Addison Corday 2007 Unearthed Sheriff Annie Flynn 2007 Saw IV Addison Corday 2008 Blonde and Blonder Cat 2008
Bachelor Party 2 : The Last Temptation Eva Direct to video 2008 Far Cry Valerie Cardinal 2009 Dolan 's Cadillac Elizabeth Mirrors 2 Elizabeth Reigns Direct to
video Hysteria Jennifer A Nanny for Christmas Ally Leeds 2011 Where the Road Meets the Sun Lisa 2011 French Immersion Jennifer Yates 2012 Stolen Child Amanda
Direct to video 2013 Absolute Deception Rebecca 2013 Susie 's Hope Donna Smith Lawrence 2014 Teen Lust Shelley 2014 Saul : The Journey to Damascus Mary
Magdalene 2015 My Stepdaughter Jill 2016 Destruction : Los Angeles Margot Taylor Television ( edit ) Year Title Role Notes 1995 A Family Divided Rosalie Frank
TV movie 1995 Highlander : The Series Maria Alcobar Episode 4.10 : `` Chivalry '' Madison Noella D'Angelo Recurring role ;
</item 3>

<item 4>
Sheila Carter
Forrester Steffy Forrester Thomas Forrester Quinn Fuller Brooke Logan Hope Logan Katie Logan Coco Spectra Sally Spectra Shirley Spectra Bill Spencer Caroline
Spencer Liam Spencer Wyatt Spencer Recurring characters Alison Montgomery Justin Barber Bradley Baker Julius Avant Vivienne Avant Saul Feinberg Antonio
Dominguez Thorne Forrester Felicia Forrester Kristen Forrester Bridget Forrester C.J. Garrison Deacon Sharpe Taylor Hayes Donna Logan Sasha Thompson Carter
Walton James Warwick Charlie Webber John Forrester Jarrett Maxwell Mateo Darlita Notable previous characters Ashley Abbott Adam Alexander Macy Alexander Morgan
DeWitt Deveney Dixon Dante Damiano Kimberly Fairchild Lauren Fenmore Aly Forrester Caroline Spencer Darla Einstein Margo Lynley Dayzee Leigh Marcus Forrester
Maggie Forrester Jessica Forrester Phoebe Forrester Stephanie Forrester Clarke Garrison Agnes Jones Whipple Jones Oliver Jones Owen Knight Beth Logan Stephen
Logan Storm Logan Erica Lovejoy Jake Maclaine Mark Maclaine Massimo Marone Jackie Marone Nick Marone Oscar Marone Amber Moore Sally Spectra Bill Spencer Karen
Spencer Danielle Spencer Production crew Executive Producer Bradley Bell Head Writers Bradley Bell Michael Minnis Creators William J.
</item 4>

<item 5>
Emmanuelle Vaugier
She has an older brother , Jason Vaugier . Career ( edit ) Television ( edit ) Vaugier played the role of young , upstart model Maria Alcobar in a 1996 episode
of Highlander : The Series . In 2004 , she played Lindsey Kellogg on the short - lived Fox Television series North Shore . She has also had guest - star spots
on Veronica Mars , Supernatural , The Outer Limits , Higher Ground , Andromeda , and Charmed . She also appeared in the Emmy Award nominated miniseries The
Beach Boys : An American Family where she portrayed Mike Love 's wife , Suzanne . Vaugier also completed a starring role in MOW Veiled Truth for the Lifetime
Network . She is also known for her role during the first two seasons of One Tree Hill , as Nicki , Jake 's ( Bryan Greenberg ) ex-girlfriend and mother to
their child , Jenny . From 2005 - 2015 , Vaugier guest starred in Twelve episodes of CBS 's hit sitcom Two and a Half Men as Mia , a ballet teacher with whom
Charlie Harper fell in love and almost married .
</item 5>

<item 6>
Krysten Ritter
The The Last International Playboy Ozzy 2008 27 Dresses Gina the goth 2008 What Happens in Vegas Kelly 2009 Glock Beretta Short film 2009 Confessions of a
Shopaholic Suze Cleath - Stuart She 's Out of My League Patty How to Make Love to a Woman Lauren 2011 Killing Bono Gloria 2011 Life Happens Kim Also writer and
co-producer 2011 Margaret Shopgirl 2012 BuzzKill Nicole 2012 Vamps Stacy Daimen 2012 Refuge Amy 2014 Listen Up Philip Melanie 2014 Veronica Mars Gia Goodman
2014 Asthma Ruby 2014 Search Party Christy 2014 Big Eyes DeAnn 2017 Hero , The The Hero Lucy Television ( edit ) Year Title Role Notes Whoopi Brynn Episode : ``
The Squatters '' One Life to Live Kay 4 episodes Law & Order Tracy Warren Episode : `` Everybody Loves Raimondo 's '' Tanner on Tanner Saleswoman 2 episodes
2005 Jonny Zero Quinn Episode : `` Pilot '' 2005 -- 2006 Veronica Mars Gia Goodman Recurring role ; 8 episodes 2006 Bedford Diaries ,
</item 6>

<item 7>
The Dukes of Hazzard (film)
Gainey as Sheriff Rosco P. Coltrane Michael Weston as Deputy Enos Strate Lynda Carter as Pauline James Roday as Billy Prickett Kevin Heffernan as Derek `` Sheev
'' Sheevington Nikki Griffin as Katie - Lynn Johnson Jacqui Maxwell as Annette Alice Greczyn as Laurie Pullman Junior Brown as The Balladeer ( narrator ) Joe
Don Baker as Governor Jim Applewhite Barry Corbin as Bill Pullman Andrew Prine as Angry Man Randy Orton as Bo Duke ( Stunt double ) Cameos ( edit ) All five
members of the comedy film troupe Broken Lizard make appearances in the film , classified as cameos , except for Kevin Heffernan , who had a larger speaking
role ( Sheev ) . Broken Lizard cameos Steve Lemme appears as Jimmy Pullman , the son of Bill Pullman , in an early car chase scene in which he accidentally
shoots the inside of his father 's truck . Jay Chandrasekhar and Erik Stolhanske reprise their roles as Ramathorn and Rabbit from the Broken Lizard comedy ,
Super Troopers .
</item 7>

<item 8>
The Divergent Series
`` I wanted to break it and say , ' It 's OK , I 'm really fun . I promise . ' But I thought , just for today , I 'd let them think that I am a complete bitch .
'' On March 25 , 2013 , Ben Lamb was cast as Edward , Ben Lloyd - Hughes as Will , and Christian Madsen as Al . In April 2013 , Ashley Judd and Tony Goldwyn
joined the cast as Natalie and Andrew Prior , Beatrice 's parents . In 2014 , the cast for the second film was announced . On May 12 , 2014 , it was announced
that Octavia Spencer joined the cast as Amity representative Johanna Reyes . Late May 2014 , Suki Waterhouse and Jonny Weston were cast as Marlene and Edgar
respectively . Early June 2014 , Stephanie Leigh Schlund announced that she was cast in the film as a member of the Amity , and Naomi Watts and Daniel Dae Kim
joined the cast as Evelyn Johnson and Jack Kang . On June 9 , 2014 , Rosa Salazar joined the cast as Lynn .
</item 8>

<item 9>
Sing (2016 American film)
Scarlett Johansson as Ash , a teenage porcupine punk rocker who takes part in an alternative - rock music duo with her boyfriend Lance . John C. Reilly as Eddie
Noodleman , a sheep and Buster 's friend who doubts the future of the theater . Tori Kelly as Meena , a teenage elephant with an exquisite singing voice and
severe stage fright . Taron Egerton as Johnny , a teenage gorilla who wants to sing , though his father would rather have him follow his criminal footsteps .
Jennifer Saunders as Nana Noodleman , a sheep and Eddie 's grandmother who was a singer in her glory days . Jennifer Hudson as Young Nana Garth Jennings as Miss
Crawly , an elderly iguana with a glass eye who is Buster 's administrative assistant . Peter Serafinowicz as Big Daddy , a gorilla gang leader who wants his
son Johnny to follow in his crime business . Nick Kroll as Gunter , a passionate dancing pig who is partnered with Rosita for the show . Beck Bennett as Lance ,
a porcupine and Ash 's boyfriend .
</item 9>

<item 10>
Emmanuelle Vaugier
Stine 's The Haunting Hour Abigail Episode 1.06 : `` The Red Dress '' 2011 Killer Mountain Kate TV movie 2012 It 's Christmas , Carol ! Carol Huffman TV movie
2013 The Mentalist Melissa Enfield Episode 5.16 : `` There Will Be Blood '' 2013 Clara 's Deadly Secret Helen TV movie 2013 Hidden Away Alexandra Cole /
Stephanie Bennett TV movie 2015 My Stepdaughter Jill TV movie 2015 Mistresses Niko Recurring role ; 6 episodes 2016 Love In Paradise Heather Twain TV movie 2016
Stranger in the House Jade TV movie 2016 His Double Life Linda TV movie 2016 Love in the Vineyard ( Heart Felt ) Molly TV movie 2017 Rogue Regan Recurring role
; 7 episodes 2017 Washed Away Gabrielle Video Games ( edit ) Year Title Role Notes 2006 Need for Speed : Carbon Nikki Voice , Mo - Cap , & Likeness 2015 Dying
Light Jade Mo - Cap 2015 Hellraid Maine , The Female Kin Voice References ( edit ) Jump up ^ `` Archived copy '' .
</item 10>
=== END ITEM id=368bbe382ce50fdc ===

=== ITEM id=f4c24cd22fd85bb3 ===
Gold sentences and facts:
Sentence 0: The previous department issued firearm for CHP officers is the Smith & Wesson Model 4006 TSW in . 40 S&W . As of August 2017 , the department has begun replacing the 4006 TSW with the Smith & Wesson M&P in . 40 S&W . Each CHP patrol vehicle is equipped with a Remington 870 Police Magnum 12 ‐ gauge shotgun and a Sig Sauer M400 rifle in 5.56 mm . Less Lethal options for officers include OC Pepper Spray and the PR - 24 Side Handle Baton . The ASP Expandable Straight Baton is authorized as an optional piece of equipment for officers completing their probationary year . Additionally , officers are authorized to carry the TASER X2 Defender . In early 2009 , officers have been allowed to mount tactical lights to their Smith & Wesson 4006 TSW pistols .
  Fact 0.0: The previous department-issued firearm for CHP officers is the Smith & Wesson Model 4006 TSW in .40 S&W.
  Fact 0.1: As of August 2017, the CHP department has begun replacing the Smith & Wesson 4006 TSW with the Smith & Wesson M&P in .40 S&W.
  Fact 0.2: Each CHP patrol vehicle is equipped with a Remington 870 Police Magnum 12-gauge shotgun.
  Fact 0.3: Each CHP patrol vehicle is equipped with a Sig Sauer M400 rifle in 5.56 mm.
  Fact 0.4: Less lethal options for CHP officers include OC Pepper Spray.
  Fact 0.5: Less lethal options for CHP officers include the PR-24 Side Handle Baton.
  Fact 0.6: The ASP Expandable Straight Baton is authorized as an optional piece of equipment for CHP officers completing their probationary year.
  Fact 0.7: CHP officers are authorized to carry the TASER X2 Defender.
  Fact 0.8: In early 2009, CHP officers were allowed to mount tactical lights to their Smith & Wesson 4006 TSW pistols.

Retrieved items:
<item 1>
California Highway Patrol
ESPN March 6 2008 Jump up ^ Game - warden shortage is about to get worse San Francisco Chronicle September 23 2007 Archived February 15 , 2012 , at the Wayback
Machine . Jump up ^ Lots of ocean , but few game wardens ! Sacramento Bee August 23 2007 Archived August 21 , 2007 , at the Wayback Machine . Jump up ^ Oregon
State Police Fish and Wildlife Division Archived 2012 - 07 - 31 at the Wayback Machine . Jump up ^ Division of Alaska Wildlife Troopers Jump up ^
http://smith-wessonforum.com/showthread.php?t=525162 ^ Jump up to : Mattos , Rick ( 2008 ) . California Highway Patrol . Charleston , SC : Arcadia Pub . ISBN
9780738556208 . Retrieved 2013 - 09 - 01 . Jump up ^ Carpenter , Susan ( 25 June 2013 ) . `` BMW 's R1200RT - P : Big , bad and packing heat '' . Los Angeles
Times . Retrieved 1 September 2013 .
</item 1>

<item 2>
California Highway Patrol
Vehicles ( edit ) This section needs expansion . You can help by adding to it . ( September 2013 ) Origins ( edit ) When motor vehicles in California were first
seen as needing legislation , law enforcement agencies began to patrol using motorcycles , cars and trucks . Motorcycle officers in 1920 Fresno , started a
group to assist each other and promote road safety - the Joaquin Valley Traffic Officer 's Association - led by Harry Wilson who they elected as president .
They renamed themselves the California Association of Highway Patrolmen in 1921 , and became the California Highway Patrol in 1927 under the auspices of the
Department of Motor Vehicles . Motorcycles ( edit ) A motorcycle sergeant on the San Tomas Expressway near Benton Avenue in Santa Clara , California . Through
the public competitive bidding process , the Harley - Davidson motorcycle was selected as the primary enforcement motorcycle for the California Highway Patrol
in 2013 . These replacement enforcement motorcycles will replenish the Department 's aging motorcycle fleet . In a cost - saving move , the CHP previously
deferred the purchase of replacement motorcycles and has not purchased enforcement motorcycles since January 14 , 2011 .
</item 2>

<item 3>
California Highway Patrol
Cases.justia.com . Retrieved 2014 - 08 - 19 . Jump up ^ ( 4 ) Jump up ^ https://www.chp.ca.gov/notify-chp/auto-theft Jump up ^ CHP Terms Explained Further
reading ( edit ) Crane , Bob , California Association of Highway Patrolmen Golden Chronicle 1920 -- 1970 , ( Sacramento , California : California Association of
Highway Patrolmen , 1970 ) . External links ( edit ) Official website CHP Traffic Incident Information Page CHP recruitment website California Association of
Highway Patrolmen ( CAHP ) website ( hide ) Highway patrol / state police in the United States See also : Law enforcement in the United States and Highway
patrol States Alabama Alaska Arizona Arkansas California Colorado Connecticut Delaware Florida Georgia Hawaii ( does not have a statewide police agency ) Idaho
Illinois Indiana Iowa Kansas Kentucky Louisiana Maine Maryland Massachusetts Michigan Minnesota Mississippi Missouri Montana Nebraska Nevada New Hampshire New
Jersey New Mexico New York North Carolina North Dakota Ohio Oklahoma Oregon Pennsylvania Rhode Island South Carolina South Dakota Tennessee Texas Utah Vermont
Virginia Washington West Virginia Wisconsin Wyoming Federal district District of Columbia Insular areas American Samoa Guam Northern Mariana Islands Puerto Rico
U.S.
</item 3>

<item 4>
California Highway Patrol
Early motorcycles used included Indian , Harley - Davidson and Henderson manufactured bikes during the 1920s and 30s ; though by 1941 the main manufacturers
used at the training academy were Harley - Davidson and Indian . In the 1970s CHP decided to renew their Harleys fleet and tested many motorcycles such as Honda
750 four , Suzuki 850 , Moto Guzzi and finally Kawasaki Z900 and then 1000C1 made famous in CHiPs . In 1998 CHP began using the BMW R1100RT - P , with features
including a helmet interface for communications , two batteries , and a top speed of 130 miles per hour ( 210 km / h ) . Vehicles ( edit ) California Highway
Patrol cruiser on display at Public Safety Day in Lakewood , California . The Department of General Services is responsible for selecting a vehicle based on
price , performance and payload capacity for the CHP . The bid specifications require a pursuit - rated , rear - wheel drive or all - wheel drive that can carry
at least 1,500 pounds , the approximate weight of four officers , their equipment , and the police vehicle equipment .
</item 4>

<item 5>
California Highway Patrol
Hierarchy ( edit ) Commissioner of the Highway Patrol - Warren A. Stanley Deputy Commissioner of the Highway Patrol - Scott Silsbee Professional Standards and
Ethics Division Office of Community Outreach & Media Relations Assistant Commissioner , Field Operations - Northern Division Valley Division Golden Gate
Division Central Division Southern Division Border Division Coastal Division Inland Division Protective Services Division Office of Air Operations State
Security Division Assistant Commissioner , Staff Operations - A . A. Browne Office of Employee Safety and Assistance Administrative Services Division
Enforcement & Planning Division Information Management Division Special Counsel to the Commissioner Field organization ( edit ) California 's 163,696 square
miles ( 423,970 square kilometers ) is divided into eight geographic areas or `` divisions '' for CHP field operations . Each division is overseen by a
commanding officer with the rank of Chief , who reports to the Assistant Commissioner of Field Operations , assisted by two or more Assistant Chiefs .
</item 5>

<item 6>
California Highway Patrol
The Ford Crown Victoria Police Interceptor averaged 16 mpg city and 24 mpg highway . The new Ford Police Interceptor Utility Vehicle costs $26,578 , which
includes a 5 - year , 100,000 - mile warranty . The Department was paying $24,043 for the Ford Crown Victoria Police Interceptors and recently spent hundreds of
thousands of dollars per month to maintain all vehicles no longer covered by warranty . Several vehicles were allocated to public affairs officers and are used
for recruiting purposes . An example of these vehicles can be seen at the California State Fair and other venues . CHP also uses Chevy Silverados , Dodge Rams (
for commercial vehicle enforcement ) , Ford Expedition , and Chevrolet Tahoe for their divisions that snow frequently or have certain terrains to permit off -
road driving . In July 2016 , it was announced that the CHP would be taking delivery of 516 2016 Dodge Chargers . In 2017 , it was announced that the CHP were
going to start using Chevrolet Tahoe Police Patrol Vehicles ( PPV ) for rural and resident posts .
</item 6>

<item 7>
California Highway Patrol
This is because of their ability to carry a larger `` basic load '' of normal issue equipment , equipment that may be only in a few cars , and / or not carried
by patrol cars at all , in a more urban setting . Aircraft ( edit ) CHP AS 350 B3 in flight . The CHP utilizes both helicopters and fixed - wing aircraft . In
2017 the fleet consisted of : 15 Airbus Helicopters ( Eurocopter ) AS350B3 Squirrels 11 Cessna T206H Turbo Stationairs 4 Gippsaero GA8 TC - 320 Airvans 2 Hughes
( MD500 ) 369Ds 1 Cessna 182S Skylane 1 Raytheon ( Beechcraft ) B300 Super King Air Rogue officers ( edit ) George Gwaltney ( edit ) A former officer from
Barstow , was convicted on May 10 , 1984 , in federal court , for the on ‐ duty rape and slaying of 23 - year - old Robin Bishop on January 11 , 1982 .
</item 7>

<item 8>
California Highway Patrol
Weapons ( edit ) This section needs additional citations for verification . Please help improve this article by adding citations to reliable sources . Unsourced
material may be challenged and removed . ( September 2014 ) ( Learn how and when to remove this template message ) The previous department issued firearm for
CHP officers is the Smith & Wesson Model 4006 TSW in . 40 S&W . As of August 2017 , the department has begun replacing the 4006 TSW with the Smith & Wesson M&P
in . 40 S&W . Each CHP patrol vehicle is equipped with a Remington 870 Police Magnum 12 ‐ gauge shotgun and a Sig Sauer M400 rifle in 5.56 mm . Less Lethal
options for officers include OC Pepper Spray and the PR - 24 Side Handle Baton . The ASP Expandable Straight Baton is authorized as an optional piece of
equipment for officers completing their probationary year . Additionally , officers are authorized to carry the TASER X2 Defender . In early 2009 , officers
have been allowed to mount tactical lights to their Smith & Wesson 4006 TSW pistols .
</item 8>

<item 9>
California Highway Patrol
Cold weather and utility uniforms are dark blue . California Highway Patrol campaign hat . Leather gear is black with brass snaps and buckles . Standard traffic
enforcement patrol vehicles are required by state law to have a white door with , in the case of the CHP , a star . The CHP operates traditional black and white
as well as all - white patrol vehicles . The California Highway Patrol is one of the few organizations to continue to use the older toll - free `` Zenith 1 ‐
2000 '' number . With the falling cost of telephone area code 800 and 888 numbers , most organizations have chosen to switch to one of the newer numbers and
discontinue use of the Zenith service , which requires operator assistance . The CHP 's traditions include its own radio codes , widely adopted by local
agencies . The most important is 11 ‐ 99 , which signifies that an officer needs emergency assistance or that an officer is down . In 1981 a charitable
foundation called the 11 ‐ 99 Foundation was founded to provide benefits and scholarships to officers and their families .
</item 9>

<item 10>
California Highway Patrol
Legal jurisdiction As per operations jurisdiction Governing body California State Legislature Constituting instrument Senate Bill 869 Headquarters Sacramento ,
California Officers 7,608 ( as of 2016 ) Civilians 3,276 ( as of 2014 ) Commissioner responsible Warren Stanley Parent agency California State Transportation
Agency ( CALSTA ) Facilities Commands 8 Field Divisions 16 Commercial Enforcement Facilities 103 Area Offices Patrol cars Ford Explorer - based Police
Interceptor Utility Vehicle Dodge Charger - Police Ford Crown Victoria - Police Planes Cessna 206 Website http://www.chp.ca.gov http://www.chpcareers.com The
California Highway Patrol ( CHP ) is a law enforcement agency of the state of California . The primary mission of the CHP is to ensure safety and enforce
traffic laws on all California highways and county roads in unincorporated areas . The CHP promotes traffic safety by inspecting commercial vehicles , as well
as inspecting and certifying school buses , ambulances , and other specialized vehicles . The California State Legislature established the California Highway
Patrol as a branch of the Division of Motor Vehicles in the Department of Public Works , with legislation signed by Governor C. C. Young on August 14 , 1929 .
</item 10>
=== END ITEM id=f4c24cd22fd85bb3 ===

=== ITEM id=74f9111647b24d1d ===
Gold sentences and facts:
Sentence 0: At the federal level in the United States , legislation ( i.e. , `` statutes '' or `` statutory law '' ) consists exclusively of Acts passed by the Congress of the United States and its predecessor , the Continental Congress , that were either signed into law by the President or passed by Congress after a presidential veto .
  Fact 0.0: At the federal level in the United States, legislation consists exclusively of Acts.
  Fact 0.1: At the federal level in the United States, legislation is also called statutes or statutory law.
  Fact 0.2: Federal legislation in the United States consists exclusively of Acts passed by the Congress of the United States.
  Fact 0.3: Federal legislation in the United States consists exclusively of Acts passed by the Continental Congress, the predecessor of the Congress of the United States.
  Fact 0.4: Acts passed by the Congress of the United States or the Continental Congress were signed into law by the President.
  Fact 0.5: Acts passed by the Congress of the United States or the Continental Congress were passed by Congress after a presidential veto.

Retrieved items:
<item 1>
State legislature (United States)
For most bills , the recommendations of the committee are followed , although either house is free to accept or reject the action of the committee . Bills
reported favorably by a committee may be placed on a regular calendar ( the agenda of the deliberative body ) . Most of the work of the legislature is done by
committees . The legislature as a whole relies on its committees to report out only those bills deserving the consideration of the entire house . Through
standing committees , each bill is addressed by a group of members who have special knowledge of its subject . Some members of the legislature have expert
knowledge of particular subjects of legislation , and these members are usually placed on committees to take full advantage of this specialized knowledge . For
this reason , the legislature often accepts the final recommendations of its standing committees . As has been noted , however , the legislature does not
completely abdicate its responsibility for the consideration of pending bills . If the need arises , the members of either house can force a committee to take
action on a bill , or they can ignore the committee 's recommendations .
</item 1>

<item 2>
List of United States federal legislation
( edit ) Acts of Congress are published in the United States Statutes at Large . Volumes 1 through 18 , which have all the statutes passed from 1789 to 1875 ,
are available on - line at the Library of Congress , here . In the list below , statutes are listed by X Stat . Y , where X is the volume of the Statutes at
Large and Y is the page number , as well as either the chapter or Public Law number . See examples below . Sessions ( Sess. . ) and Chapters ( ch. . ) ( edit )
Each Congress has two to four sessions . Under the numbering system used from 1789 until 1957 , the Acts in each session are numbered sequentially as Chapters .
This numbering included both laws applicable to the general public and laws relating to specific individuals , e.g. , to grant pensions to disabled veterans .
Public laws ( edit ) Since 1957 , Acts of Congress related to the general public have been designated in the form : Public Law X-Y where X is the number of the
ordinal Congress and Y is the number of the chronological order of the public Act in that Congress .
</item 2>

<item 3>
State legislature (United States)
If the majority vote in favor of the bill , it is recorded as passed . Transmission to second house ( edit ) A bill that is passed in one house is transmitted ,
along with a formal message , to the other house . If the bill is not reported from committee or is not considered by the full house , the bill is defeated .
The house of origin , upon return of its amended bill , may take any one of several courses of action . It may concur in the amendment by the adoption of a
motion to that effect ; then the bill , having been passed by both houses in identical form , is ready for enrollment . Another possibility is that the house of
origin may adopt a motion to non-concur in the amendment , at which point the bill dies . Finally , the house of origin may refuse to accept the amendment but
request that a conference committee be appointed . The other house usually agrees to the request , and the presiding officer of each house appoints members to
the conference committee .
</item 3>

<item 4>
List of United States federal legislation
However , most executive branch and Judicial Branch regulations must originate in a congressional grant of power . See also : Executive orders of the President
; regulations of Executive branch departments and administrative agencies ; and the procedural rules of the federal courts . Contents ( hide ) 1 Publication of
the law 1.1 Statutes at Large ( Stat . ) 1.2 Sessions ( Sess . ) and Chapters ( ch . ) 1.3 Public laws 1.4 Examples 2 Congress of the Confederation 3 United
States Congress 3.1 1789 to 1901 : 1st through 56th Congresses 3.2 1901 to 2001 : 57th through 106th Congresses 3.3 2001 to present : 107th and subsequent
Congresses 4 See also 5 Sources 6 References Publication of the law ( edit ) Statutes at Large ( Stat. . ) ( edit ) Acts of Congress are published in the United
States Statutes at Large .
</item 4>

<item 5>
State legislature (United States)
If the governor does not approve of the bill , he or she may veto it . In the event of a veto , the governor returns the bill to the house in which it
originated , with a message explaining his or her objections and suggesting any amendments ( if applicable ) which might remove those objections . The bill is
then reconsidered , and if a simple majority of the members of both houses agrees to the proposed executive amendments , it is returned to the Governor , as he
or she revised it , for his or her signature . On the other hand , a simple majority of the members of each house can choose to approve a vetoed bill precisely
as the Legislature originally passed it , in which case it becomes a law over the governor 's veto . This is in contrast to the practice in most states and the
federal government , which require a two - thirds majority in both houses to override a governor 's veto . If the Governor fails to return a bill to the
legislative house in which it originated within a specified number of days after it was presented to him or her , it becomes a law without their signature .
</item 5>

<item 6>
List of United States federal legislation
MacMillan Reference Books ( December 2003 ) ISBN 0 - 02 - 865749 - 7 References ( edit ) Jump up ^ See , e.g. , Act of Aug. 11 , 1790 , ch. 44 , 6 Stat. 3 Jump
up ^ About Bills , Resolutions , and Laws United States Congress House of Representatives Senate ( 114th ← 115th → 116th ) Lists of congressional lists Members
and leaders Members Current By length of service Freshmen Youngest members Resident Commissioner of Puerto Rico Delegates Senate Current by seniority Dean of
the Senate Former U.S. Senators living Earliest serving Earliest living Expelled or censured Classes Born outside the U.S. House Current by seniority Dean of
the House Former U.S. Representatives living Oldest living Earliest serving Expelled , censured ,
</item 6>

<item 7>
List of United States federal legislation
This numbering is continuous across sessions of Congress and does not include acts relating to individuals , which are analogously given Private Law X-Y
designations . ( From 1901 to 1957 , the Statutes at Large listed both a chapter and a Public / Private number for each act ) See examples below. There are many
laws . Examples ( edit ) The Militia Act of 1862 of July 17 , 1862 , Sess. 2 , ch. 201 , was the 201st Act of the second session of the 37th Congress . The
National Banking Act of February 25 , 1863 , Sess. 3 , ch. 58 , was the 58th Act of the third session of the 37th Congress . The Global Anti-Semitism Review Act
of 2004 of October 16 , 2004 , Pub. L. 108 -- 332 , 118 Stat. 1282 , was the 332nd Act of Congress ( statute ) passed in the 108th Congress . It can be found in
volume 118 of the U.S. Statutes at Large , starting at page 1282 .
</item 7>

<item 8>
List of United States federal legislation
List of United States federal legislation - wikipedia List of United States federal legislation Jump to : navigation , search This article includes a list of
references , but its sources remain unclear because it has insufficient inline citations . Please help to improve this article by introducing more precise
citations. ( April 2016 ) ( Learn how and when to remove this template message ) See also : Category : United States federal legislation This is a chronological
, but still incomplete , list of United States federal legislation . Congress has enacted approximately 200 -- 600 statutes during each of its 115 biennial
terms , so that more than 20,000 statutes have been enacted since 1789 . At the federal level in the United States , legislation ( i.e. , `` statutes '' or ``
statutory law '' ) consists exclusively of Acts passed by the Congress of the United States and its predecessor , the Continental Congress , that were either
signed into law by the President or passed by Congress after a presidential veto . Legislation is not the only source of regulations with the force of law .
However , most executive branch and Judicial Branch regulations must originate in a congressional grant of power .
</item 8>

<item 9>
State legislature (United States)
Conference committees ( edit ) A conference committee is often empaneled to discuss the points of difference between the two houses ' versions of the same bill
, and tries to reach an agreement between them so that the identical bill can be passed by both houses . If an agreement is reached and if both houses adopt the
conference committee 's report , the bill is passed . If either house refuses to adopt the report of the conference committee , a motion may be made for further
conference . If a conference committee is unable to reach an agreement , it may be discharged , and a new conference committee may be appointed . Some highly
controversial bills may be referred to several different conference committees . If an agreement is never reached in conference prior to the end of the
legislative session , the bill is lost . When a bill has passed both houses in identical form , it is then ready for transmittal to the Governor . Presentation
to the governor ( edit ) Once a bill reaches the governor , he or she may sign it , which completes its enactment into law . From this point , the bill becomes
an act , and remains the law of the state unless repealed by legislative action , or overturned by a court decision .
</item 9>

<item 10>
State legislature (United States)
Upon introduction , a bill is usually read by its title only , constituting the first reading of the bill . Because a bill is usually read by title only , it is
important that the title give the members notice of the subject matter contained in the bill . Committees ( edit ) General information ( edit ) As with other
legislative bodies throughout the world , U.S. state legislatures operate mainly through committees when considering proposed bills . Thus , committee action is
probably the most important phase of the legislative process . Most bills can not be enacted into law until it has been referred to , acted upon by , and
returned from , a standing committee in each house . Reference to committee usually follows the first reading of the bill . Each committee is set up to consider
bills relating to a particular subject . Standing committees are charged with the important responsibility of examining bills and recommending action to the
Senate or House . Often on days when a legislature is not in session , the committees of each house meet and consider the bills that have been referred to them
to decide if the assigned bills should be reported for further action .
</item 10>
=== END ITEM id=74f9111647b24d1d ===
